"""Main application window for the mark desktop automation agent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading

import customtkinter as ctk

from agent.callbacks import AgentCallbacks
from agent.config import MarkConfig
from agent.loop import AgentLoop
from agent.mcp_client import connect_mcp
from agent.orchestrator import Orchestrator
from ui.result_card import ResultCard
from ui.step_card import StepCard
from ui.theme import (
    BG, BORDER, FG, FG_DIM, FG_FAINT, FG_GREEN, FG_RED,
    HIGHLIGHT, MODELS, MODEL_PRICES, MONO,
    PAD, REASONING_LEVELS, SURFACE, WIDTH,
    trunc, unwrap_exc,
)

logger = logging.getLogger(__name__)


class MarkApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()

        self._agent_thread: threading.Thread | None = None
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._callbacks: AgentCallbacks | None = None
        self._guidance_text: str | None = None
        self._lock = threading.Lock()
        self._step_cards: dict[tuple[int, int], StepCard] = {}
        self._current_goal: int = 0
        self._running = False
        self._paused = False
        self._result_card: ResultCard | None = None
        self._pending_plan: list[str] | None = None

        self._setup_window()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.createcommand("::tk::mac::ReopenApplication", self._reopen)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.title("")
        self.configure(fg_color=BG)
        sw = self.winfo_screenwidth()
        self.geometry(f"{WIDTH}x740+{sw - WIDTH - 20}+50")
        self.minsize(WIDTH, 500)
        self.attributes("-topmost", True)

    def _fit_height(self) -> None:
        req = self.winfo_reqheight()
        screen_h = self.winfo_screenheight()
        h = max(500, min(req, screen_h - 80))
        x, y = self.winfo_x(), self.winfo_y()
        self.geometry(f"{WIDTH}x{h}+{x}+{y}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        c = ctk.CTkFrame(self, fg_color="transparent")
        c.pack(fill="both", expand=True, padx=PAD, pady=(PAD, 1))

        self._add_header(c)
        self._add_config(c)
        self._add_task(c)
        self._add_status(c)
        self._add_steps(c)
        self._add_guidance(c)

    def _add_header(self, p: ctk.CTkBaseClass) -> None:
        hf = ctk.CTkFrame(p, fg_color="transparent")
        hf.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            hf, text="\u25b8",
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG_FAINT, anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            hf, text="mark",
            font=ctk.CTkFont(family=MONO, size=16, weight="bold"),
            text_color=FG, anchor="w",
        ).pack(side="left", padx=(2, 0))
        ctk.CTkLabel(
            hf, text="desktop agent",
            font=ctk.CTkFont(family=MONO, size=12),
            text_color=FG_FAINT, anchor="w",
        ).pack(side="left", padx=(6, 0))

    # -- configuration --

    def _add_config(self, p: ctk.CTkBaseClass) -> None:
        self._cfg_open = False
        self._cfg_btn = ctk.CTkButton(
            p, text="+ config",
            command=self._toggle_config,
            fg_color="transparent", text_color=FG_DIM,
            hover_color=HIGHLIGHT, anchor="w", height=18,
            font=ctk.CTkFont(family=MONO, size=13),
            corner_radius=0,
        )
        self._cfg_btn.pack(fill="x", pady=(0, 2))

        self._cfg_frame = ctk.CTkFrame(
            p, corner_radius=0, fg_color=SURFACE,
            border_width=1, border_color=BORDER,
        )
        self._cfg_frame.columnconfigure(1, weight=1)

        lbl_kw = dict(
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG_DIM,
        )
        entry_kw = dict(
            fg_color=BG, border_color=BORDER,
            text_color=FG, corner_radius=0,
        )

        r = 0
        ctk.CTkLabel(self._cfg_frame, text="  api key", **lbl_kw).grid(
            row=r, column=0, sticky="w", padx=0, pady=1,
        )
        self._api_key_var = ctk.StringVar(
            master=self, value=os.environ.get("OPENAI_API_KEY", ""),
        )
        self._api_key_entry = ctk.CTkEntry(
            self._cfg_frame,
            textvariable=self._api_key_var,
            width=150, height=26, show="\u2022",
            font=ctk.CTkFont(family=MONO, size=13),
            **entry_kw,
        )
        self._api_key_entry.grid(row=r, column=1, padx=PAD, pady=1, sticky="e")

        r += 1
        ctk.CTkLabel(self._cfg_frame, text="  exec model", **lbl_kw).grid(
            row=r, column=0, sticky="w", padx=0, pady=1,
        )
        self._model_var = ctk.StringVar(master=self, value="gpt-5-nano")
        ctk.CTkComboBox(
            self._cfg_frame, values=MODELS,
            variable=self._model_var,
            width=150, height=26,
            font=ctk.CTkFont(family=MONO, size=13),
            dropdown_font=ctk.CTkFont(family=MONO, size=13),
            button_color=BORDER, button_hover_color=FG_FAINT,
            dropdown_fg_color=SURFACE,
            **entry_kw,
        ).grid(row=r, column=1, padx=PAD, pady=1, sticky="e")

        r += 1
        ctk.CTkLabel(self._cfg_frame, text="  plan model", **lbl_kw).grid(
            row=r, column=0, sticky="w", padx=0, pady=1,
        )
        self._plan_model_var = ctk.StringVar(master=self, value="gpt-5-nano")
        ctk.CTkComboBox(
            self._cfg_frame, values=MODELS,
            variable=self._plan_model_var,
            width=150, height=26,
            font=ctk.CTkFont(family=MONO, size=13),
            dropdown_font=ctk.CTkFont(family=MONO, size=13),
            button_color=BORDER, button_hover_color=FG_FAINT,
            dropdown_fg_color=SURFACE,
            **entry_kw,
        ).grid(row=r, column=1, padx=PAD, pady=1, sticky="e")

        r += 1
        self._price_lbl = ctk.CTkLabel(
            self._cfg_frame, text="",
            font=ctk.CTkFont(family=MONO, size=11),
            text_color=FG_FAINT, anchor="e",
        )
        self._price_lbl.grid(row=r, column=0, columnspan=2, padx=PAD, pady=0, sticky="e")
        self._model_var.trace_add("write", lambda *_: self._update_price())
        self._update_price()

        r += 1
        ctk.CTkLabel(self._cfg_frame, text="  reasoning", **lbl_kw).grid(
            row=r, column=0, sticky="w", padx=0, pady=1,
        )
        self._reason_var = ctk.StringVar(master=self, value=MarkConfig.reasoning_effort)
        ctk.CTkComboBox(
            self._cfg_frame, values=REASONING_LEVELS,
            variable=self._reason_var,
            width=150, height=26,
            font=ctk.CTkFont(family=MONO, size=13),
            dropdown_font=ctk.CTkFont(family=MONO, size=13),
            button_color=BORDER, button_hover_color=FG_FAINT,
            dropdown_fg_color=SURFACE,
            **entry_kw,
        ).grid(row=r, column=1, padx=PAD, pady=1, sticky="e")

        r += 1
        ctk.CTkLabel(self._cfg_frame, text="  temp", **lbl_kw).grid(
            row=r, column=0, sticky="w", padx=0, pady=1,
        )
        tf = ctk.CTkFrame(self._cfg_frame, fg_color="transparent")
        tf.grid(row=r, column=1, padx=PAD, pady=1, sticky="e")
        self._temp_var = ctk.DoubleVar(master=self, value=MarkConfig.temperature)
        self._temp_lbl = ctk.CTkLabel(
            tf, text=f"{MarkConfig.temperature:.2f}",
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG, width=32,
        )
        self._temp_lbl.pack(side="right")
        ctk.CTkSlider(
            tf, from_=0, to=1, variable=self._temp_var,
            command=lambda v: self._temp_lbl.configure(text=f"{v:.2f}"),
            width=110, height=12,
            button_color=FG_DIM, button_hover_color=FG,
            progress_color=FG_FAINT, corner_radius=0,
        ).pack(side="right", padx=(0, 6))

        r += 1
        sw_frame = ctk.CTkFrame(self._cfg_frame, fg_color="transparent")
        sw_frame.grid(row=r, column=0, columnspan=2, padx=PAD, pady=1, sticky="ew")
        sw_frame.columnconfigure(0, weight=1)
        sw_frame.columnconfigure(1, weight=1)

        sw_kw = dict(
            height=20,
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG_DIM, progress_color=FG_DIM,
            button_color=FG, button_hover_color=FG,
        )

        self._vision_var = ctk.BooleanVar(master=self, value=True)
        ctk.CTkSwitch(
            sw_frame, text="vision", variable=self._vision_var,
            command=self._on_vision_toggle, **sw_kw,
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))

        self._orch_var = ctk.BooleanVar(master=self, value=True)
        ctk.CTkSwitch(sw_frame, text="orchestrator", variable=self._orch_var, **sw_kw
        ).grid(row=0, column=1, sticky="w", pady=(0, 2))

        self._plan_edit_var = ctk.BooleanVar(master=self, value=True)
        ctk.CTkSwitch(sw_frame, text="plan edit", variable=self._plan_edit_var, **sw_kw
        ).grid(row=1, column=0, sticky="w")

        self._debug_var = ctk.BooleanVar(master=self, value=False)
        ctk.CTkSwitch(sw_frame, text="debug logs", variable=self._debug_var, **sw_kw
        ).grid(row=1, column=1, sticky="w")

        self._omniparser_var = ctk.BooleanVar(master=self, value=False)
        self._omniparser_switch = ctk.CTkSwitch(
            sw_frame, text="omniparser", variable=self._omniparser_var,
            **sw_kw,
        )
        self._omniparser_switch.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        r += 1
        self._adv_open = False
        self._adv_btn = ctk.CTkButton(
            self._cfg_frame, text="  + advanced",
            command=self._toggle_adv,
            fg_color="transparent", text_color=FG_FAINT,
            hover_color=HIGHLIGHT, anchor="w", height=20,
            font=ctk.CTkFont(family=MONO, size=13),
            corner_radius=0,
        )
        self._adv_btn.grid(row=r, column=0, columnspan=2, padx=0, pady=(2, 0), sticky="w")

        r += 1
        self._adv_frame = ctk.CTkFrame(self._cfg_frame, fg_color="transparent")
        self._adv_row = r

        adv_defs = [
            ("llm_timeout", "llm_timeout"),
            ("mcp_timeout", "mcp_timeout"),
            ("max_failures", "max_failures"),
            ("max_stale_steps", "max_stale"),
            ("step_delay", "step_delay"),
            ("post_action_delay", "action_delay"),
            ("max_messages", "max_messages"),
            ("max_goals", "max_goals"),
        ]
        self._adv_vars: dict[str, ctk.StringVar] = {}
        for i, (key, label) in enumerate(adv_defs):
            default = str(MarkConfig.__dataclass_fields__[key].default)
            ctk.CTkLabel(
                self._adv_frame, text=f"  {label}",
                font=ctk.CTkFont(family=MONO, size=13),
                text_color=FG_FAINT,
            ).grid(row=i, column=0, sticky="w", pady=1)
            v = ctk.StringVar(master=self, value=default)
            ctk.CTkEntry(
                self._adv_frame, textvariable=v,
                width=70, height=24,
                font=ctk.CTkFont(family=MONO, size=13),
                **entry_kw,
            ).grid(row=i, column=1, padx=PAD, pady=1, sticky="e")
            self._adv_vars[key] = v

    def _task_entry_return(self, event) -> str:
        self._run_agent()
        return "break"  # prevent newline insertion

    def _toggle_config(self) -> None:
        self._cfg_open = not self._cfg_open
        if self._cfg_open:
            self._cfg_frame.pack(after=self._cfg_btn, fill="x", pady=(0, 2))
            self._cfg_btn.configure(text="- config")
        else:
            self._cfg_frame.pack_forget()
            self._cfg_btn.configure(text="+ config")

    def _update_price(self) -> None:
        prices = MODEL_PRICES.get(self._model_var.get())
        if prices:
            inp, out = prices
            self._price_lbl.configure(text=f"${inp:g} in \u00b7 ${out:g} out /MTok")
        else:
            self._price_lbl.configure(text="")

    def _on_vision_toggle(self) -> None:
        if self._vision_var.get():
            self._omniparser_switch.grid(
                row=2, column=0, columnspan=2, sticky="w", pady=(2, 0),
            )
        else:
            self._omniparser_var.set(False)
            self._omniparser_switch.grid_forget()

    def _toggle_adv(self) -> None:
        self._adv_open = not self._adv_open
        if self._adv_open:
            self._adv_frame.grid(
                row=self._adv_row, column=0, columnspan=2,
                padx=0, pady=(1, PAD), sticky="ew",
            )
            self._adv_btn.configure(text="  - advanced")
        else:
            self._adv_frame.grid_forget()
            self._adv_btn.configure(text="  + advanced")

    # -- task + controls --

    def _add_task(self, p: ctk.CTkBaseClass) -> None:
        tf = ctk.CTkFrame(p, fg_color="transparent")
        tf.pack(fill="x", pady=(0, 3))

        ctk.CTkLabel(
            tf, text="task>",
            font=ctk.CTkFont(family=MONO, size=12),
            text_color=FG_FAINT, anchor="w", width=36,
        ).pack(side="left")

        self._task_entry = ctk.CTkTextbox(
            tf,
            height=28, fg_color=SURFACE,
            border_color=BORDER, border_width=2, text_color=FG,
            font=ctk.CTkFont(family=MONO, size=13),
            corner_radius=0, wrap="word",
            activate_scrollbars=False,
        )
        self._task_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._task_entry.bind("<Return>", self._task_entry_return)

        bf = ctk.CTkFrame(p, fg_color="transparent")
        bf.pack(fill="x", pady=(0, 2))
        bf.columnconfigure(0, weight=3)
        bf.columnconfigure(1, weight=2)
        bf.columnconfigure(2, weight=2)

        btn_kw = dict(
            height=28, corner_radius=0,
            font=ctk.CTkFont(family=MONO, size=13),
            border_width=1, border_color=BORDER,
        )

        self._run_btn = ctk.CTkButton(
            bf, text="[ run ]", command=self._run_agent,
            fg_color=SURFACE, hover_color=HIGHLIGHT,
            text_color=FG, **btn_kw,
        )
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self._pause_btn = ctk.CTkButton(
            bf, text="[ pause ]", command=self._pause_resume,
            fg_color=BG, hover_color=HIGHLIGHT,
            text_color=FG_DIM, state="disabled", **btn_kw,
        )
        self._pause_btn.grid(row=0, column=1, sticky="ew", padx=2)

        self._stop_btn = ctk.CTkButton(
            bf, text="[ stop ]", command=self._stop_agent,
            fg_color=BG, hover_color=HIGHLIGHT,
            text_color=FG_DIM, state="disabled", **btn_kw,
        )
        self._stop_btn.grid(row=0, column=2, sticky="ew", padx=(2, 0))

    # -- status --

    def _add_status(self, p: ctk.CTkBaseClass) -> None:
        sf = ctk.CTkFrame(p, fg_color="transparent")
        sf.pack(fill="x", pady=(2, 1))

        self._status = ctk.CTkLabel(
            sf, text="> idle",
            font=ctk.CTkFont(family=MONO, size=14),
            text_color=FG_DIM, anchor="w",
        )
        self._status.pack(fill="x")

        self._goal_lbl = ctk.CTkLabel(
            sf, text="",
            font=ctk.CTkFont(family=MONO, size=13),
            text_color=FG_FAINT, anchor="w",
            wraplength=WIDTH - 30,
        )
        self._goal_lbl.pack(fill="x")

        # Plan editor panel (hidden by default, shown when allow_plan_edit is on)
        self._plan_editor = ctk.CTkFrame(sf, fg_color="transparent")

        self._goals_frame = ctk.CTkFrame(self._plan_editor, fg_color="transparent")
        self._goals_frame.pack(fill="x", pady=(2, 2))
        self._goal_entries: list[ctk.CTkTextbox] = []

        refine_row = ctk.CTkFrame(self._plan_editor, fg_color="transparent")
        refine_row.pack(fill="x", pady=(0, 2))

        self._refine_entry = ctk.CTkEntry(
            refine_row, placeholder_text="refine: ...",
            height=24, fg_color=SURFACE,
            border_color=BORDER, text_color=FG,
            font=ctk.CTkFont(family=MONO, size=13),
            corner_radius=0,
        )
        self._refine_entry.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self._refine_btn = ctk.CTkButton(
            refine_row, text="[ refine ]",
            command=self._refine_plan,
            height=24, width=70,
            font=ctk.CTkFont(family=MONO, size=13),
            fg_color=SURFACE, hover_color=HIGHLIGHT,
            text_color=FG_DIM, corner_radius=0,
            border_width=1, border_color=BORDER,
        )
        self._refine_btn.pack(side="right")

        self._confirm_plan_btn = ctk.CTkButton(
            self._plan_editor, text="[ confirm plan ]",
            command=self._confirm_plan,
            height=26,
            font=ctk.CTkFont(family=MONO, size=13),
            fg_color=SURFACE, hover_color=HIGHLIGHT,
            text_color=FG_GREEN, corner_radius=0,
            border_width=1, border_color=FG_GREEN,
        )
        self._confirm_plan_btn.pack(fill="x", pady=(0, 2))

    # -- steps --

    def _add_steps(self, p: ctk.CTkBaseClass) -> None:
        self._steps_frame = ctk.CTkScrollableFrame(
            p, fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=FG_FAINT,
        )
        self._steps_frame.pack(fill="both", expand=True, pady=(0, 2))

    def _scroll_steps_down(self) -> None:
        def _scroll() -> None:
            try:
                self._steps_frame._parent_canvas.yview_moveto(1.0)
            except AttributeError:
                pass
        self.after(60, _scroll)

    # -- guidance --

    def _add_guidance(self, p: ctk.CTkBaseClass) -> None:
        gf = ctk.CTkFrame(p, fg_color="transparent")
        gf.pack(fill="x", pady=(2, 0))

        ctk.CTkLabel(
            gf, text="guide>",
            font=ctk.CTkFont(family=MONO, size=12),
            text_color=FG_FAINT, anchor="w", width=42,
        ).pack(side="left")

        self._guide_entry = ctk.CTkEntry(
            gf, placeholder_text="",
            height=24, fg_color=SURFACE,
            border_color=BORDER, text_color=FG,
            font=ctk.CTkFont(family=MONO, size=13),
            corner_radius=0,
        )
        self._guide_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._guide_entry.bind("<Return>", lambda _e: self._send_guidance())

        ctk.CTkButton(
            gf, text=">>",
            command=self._send_guidance,
            width=30, height=24,
            font=ctk.CTkFont(family=MONO, size=13),
            fg_color=SURFACE, hover_color=HIGHLIGHT,
            text_color=FG_DIM, corner_radius=0,
            border_width=1, border_color=BORDER,
        ).pack(side="right")

        self._guide_status = ctk.CTkLabel(
            p, text="",
            font=ctk.CTkFont(family=MONO, size=11),
            text_color=FG_FAINT, anchor="w", height=14,
        )
        self._guide_status.pack(fill="x")

    # ==================================================================
    # Agent control
    # ==================================================================

    def _build_config_obj(self) -> MarkConfig:
        kw: dict = {
            "model": self._model_var.get() or None,
            "orchestrator_model": self._plan_model_var.get() or None,
            "temperature": self._temp_var.get(),
            "reasoning_effort": self._reason_var.get(),
            "send_images": self._vision_var.get(),
            "allow_plan_edit": self._plan_edit_var.get(),
            "save_debug_logs": self._debug_var.get(),
            "use_omniparser": self._omniparser_var.get(),
        }
        for key, var in self._adv_vars.items():
            raw = var.get()
            try:
                default = MarkConfig.__dataclass_fields__[key].default
                kw[key] = type(default)(raw)
            except (ValueError, KeyError, TypeError):
                pass
        return MarkConfig(**kw)

    def _run_agent(self) -> None:
        task = self._task_entry.get("1.0", "end-1c").strip()
        if not task or self._running:
            return

        api_key = self._api_key_var.get().strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        for card in self._step_cards.values():
            card.destroy()
        self._step_cards.clear()
        self._current_goal = 0

        if self._result_card is not None:
            self._result_card.destroy()
            self._result_card = None
        self._pending_plan = None
        self._plan_editor.pack_forget()

        self._running = True
        self._paused = False
        self._run_btn.configure(state="disabled")
        self._pause_btn.configure(state="normal", text="[ pause ]")
        self._stop_btn.configure(state="normal")
        self._task_entry.configure(state="disabled", text_color=FG_DIM)
        self._status.configure(text="> starting...", text_color=FG)
        self._goal_lbl.configure(text="")

        config = self._build_config_obj()
        use_orch = self._orch_var.get()

        self._callbacks = AgentCallbacks(
            on_step_start=lambda s: self.after(0, self._ui_step_start, s),
            on_perceive_start=lambda s, omni: self.after(0, self._ui_perceive_start, s, omni),
            on_think=lambda s, o, t, a: self.after(0, self._ui_think, s, o, t, a),
            on_action_result=lambda s, n, r: self.after(0, self._ui_action_result, s, n, r),
            on_goal_start=lambda i, t, g: self.after(0, self._ui_goal_start, i, t, g),
            on_goal_end=lambda i, t, r: self.after(0, self._ui_goal_end, i, t, r),
            on_decompose=lambda g: self.after(0, self._ui_decompose, g),
            on_done=lambda r: self.after(0, self._ui_done, r),
            get_guidance=self._consume_guidance,
            get_plan=self._consume_plan,
        )

        self._agent_thread = threading.Thread(
            target=self._thread_main,
            args=(task, config, use_orch),
            daemon=True,
        )
        self._agent_thread.start()

    def _thread_main(self, task: str, config: MarkConfig, use_orch: bool) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._agent_loop = loop
        try:
            loop.run_until_complete(self._async_run(task, config, use_orch))
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            logger.exception("Agent error")
            self.after(0, self._ui_error, unwrap_exc(exc))
        finally:
            self._agent_loop = None
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True),
                    )
            except Exception:
                pass
            finally:
                loop.close()
            self.after(0, self._ui_finished)

    async def _async_run(self, task: str, config: MarkConfig, use_orch: bool) -> None:
        if config.use_omniparser:
            url = f"http://127.0.0.1:{config.omniparser_port}"
            os.environ["OMNIPARSER_LOCAL_URL"] = url
        async with connect_mcp(
            "vision", sys.executable, ["-m", "servers.vision.server"],
        ) as vision:
            async with connect_mcp(
                "action", sys.executable, ["-m", "servers.action.server"],
            ) as action:
                if use_orch:
                    runner = Orchestrator(
                        task, config, vision, action,
                        callbacks=self._callbacks,
                    )
                else:
                    runner = AgentLoop(
                        task, config, vision, action,
                        callbacks=self._callbacks,
                    )
                await runner.run()

    def _pause_resume(self) -> None:
        if not self._callbacks:
            return
        if self._paused:
            self._callbacks.pause_event.set()
            self._paused = False
            self._pause_btn.configure(text="[ pause ]")
            self._status.configure(text="> running...", text_color=FG)
        else:
            self._callbacks.pause_event.clear()
            self._paused = True
            self._pause_btn.configure(text="[ resume ]")
            self._status.configure(text="> paused", text_color=FG_DIM)

    def _stop_agent(self) -> None:
        if not self._callbacks:
            return
        self._callbacks.stop_requested = True
        self._callbacks.pause_event.set()
        loop = self._agent_loop
        if loop and not loop.is_closed():
            loop.call_soon_threadsafe(self._cancel_agent_tasks)
        self._status.configure(text="> stopping...", text_color=FG_RED)

    def _cancel_agent_tasks(self) -> None:
        loop = self._agent_loop
        if loop:
            for task in asyncio.all_tasks(loop):
                task.cancel()

    # -- guidance --

    def _send_guidance(self) -> None:
        text = self._guide_entry.get().strip()
        if not text:
            return
        with self._lock:
            self._guidance_text = text
        self._guide_entry.delete(0, "end")
        self._guide_status.configure(
            text=f"  queued: {trunc(text, 45)}",
            text_color=FG_DIM,
        )

    def _consume_guidance(self) -> str | None:
        with self._lock:
            text = self._guidance_text
            self._guidance_text = None
        if text is not None:
            self.after(
                0,
                lambda: self._guide_status.configure(text="", text_color=FG_FAINT),
            )
        return text

    def _consume_plan(self) -> list[str] | None:
        with self._lock:
            plan = self._pending_plan
            self._pending_plan = None
        return plan

    # -- plan editor --

    def _populate_goal_boxes(self, goals: list[str]) -> None:
        for child in self._goals_frame.winfo_children():
            child.destroy()
        self._goal_entries.clear()
        for i, goal in enumerate(goals, 1):
            row = ctk.CTkFrame(self._goals_frame, fg_color="transparent")
            row.pack(fill="x", pady=(0, 3))
            ctk.CTkLabel(
                row, text=f"{i}.",
                font=ctk.CTkFont(family=MONO, size=13),
                text_color=FG_FAINT, width=20, anchor="nw",
            ).pack(side="left", anchor="n", padx=(0, 4))
            tb = ctk.CTkTextbox(
                row, height=52,
                fg_color=SURFACE, border_color=BORDER, border_width=1,
                text_color=FG,
                font=ctk.CTkFont(family=MONO, size=13),
                corner_radius=0, wrap="word",
            )
            tb.pack(fill="x", expand=True)
            tb.insert("1.0", goal)
            self._goal_entries.append(tb)

    def _read_goal_boxes(self) -> list[str]:
        goals = []
        for tb in self._goal_entries:
            text = tb.get("1.0", "end").strip()
            if text:
                goals.append(re.sub(r"^\d+\.\s*", "", text))
        return goals

    def _confirm_plan(self) -> None:
        goals = self._read_goal_boxes()
        with self._lock:
            if goals:
                self._pending_plan = goals
        self._plan_editor.pack_forget()
        if self._callbacks is not None:
            self._callbacks.plan_confirm_event.set()

    def _refine_plan(self) -> None:
        goals = self._read_goal_boxes()
        prompt_text = self._refine_entry.get().strip()
        if not prompt_text or not goals:
            return
        self._refine_btn.configure(state="disabled", text="[ refining... ]")

        def _do_refine() -> None:
            import asyncio as _asyncio
            from agent.llm import OpenAILLM
            from agent.orchestrator import GoalDecomposition
            config = self._build_config_obj()
            llm = OpenAILLM(config)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a task planner. Given a list of goals and a "
                        "refinement instruction, return an updated JSON with field "
                        "'goals' as a list of strings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current goals:\n"
                        + "\n".join(f"{i+1}. {g}" for i, g in enumerate(goals))
                        + f"\n\nRefinement: {prompt_text}"
                    ),
                },
            ]
            try:
                result = _asyncio.run(llm.decide(messages, GoalDecomposition))
                self.after(0, self._on_refine_done, result.goals)
            except Exception:
                self.after(0, self._on_refine_done, None)

        threading.Thread(target=_do_refine, daemon=True).start()

    def _on_refine_done(self, goals: list[str] | None) -> None:
        self._refine_btn.configure(state="normal", text="[ refine ]")
        if goals:
            self._populate_goal_boxes(goals)
        self._refine_entry.delete(0, "end")

    # ==================================================================
    # UI callbacks (main thread)
    # ==================================================================

    def _card_key(self, step: int) -> tuple[int, int]:
        return (self._current_goal, step)

    def _ui_perceive_start(self, step: int, omniparser: bool) -> None:
        if omniparser:
            self._status.configure(
                text=f"> step {step}: parsing screen (omniparser)...",
                text_color=FG,
            )

    def _ui_step_start(self, step: int) -> None:
        for card in self._step_cards.values():
            card.collapse()
        key = self._card_key(step)
        card = StepCard(self._steps_frame, step, goal_idx=self._current_goal)
        card.pack(fill="x", pady=(0, 2))
        self._step_cards[key] = card
        self._status.configure(text=f"> step {step}...", text_color=FG)
        self._scroll_steps_down()

    def _ui_think(self, step: int, obs: str, thought: str, actions: list[dict]) -> None:
        card = self._step_cards.get(self._card_key(step))
        if card:
            card.set_thinking(obs, thought, actions)
        self._scroll_steps_down()

    def _ui_action_result(self, step: int, name: str, result: dict) -> None:
        card = self._step_cards.get(self._card_key(step))
        if card:
            card.add_result(name, result)
            if not result.get("success", False):
                card.set_status("failed")
        self._scroll_steps_down()

    def _ui_goal_start(self, idx: int, total: int, goal: str) -> None:
        stale = [k for k in self._step_cards if k[0] == idx]
        for k in stale:
            self._step_cards.pop(k).destroy()
        self._current_goal = idx
        self._status.configure(text=f"> goal {idx}/{total}", text_color=FG)
        self._goal_lbl.configure(text=f"  \u25b8 {trunc(goal, 100)}")

        sep_frame = ctk.CTkFrame(
            self._steps_frame, fg_color="transparent",
        )
        sep_frame.pack(fill="x", pady=(5, 2))
        ctk.CTkLabel(
            sep_frame,
            text=f"goal {idx}/{total}",
            font=ctk.CTkFont(family=MONO, size=12, weight="bold"),
            text_color=FG_DIM, anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            sep_frame,
            text=f"  {trunc(goal, 55)}",
            font=ctk.CTkFont(family=MONO, size=12),
            text_color=FG_FAINT, anchor="w",
        ).pack(side="left")
        self._scroll_steps_down()

    def _ui_goal_end(self, idx: int, total: int, result: str) -> None:
        for key, card in self._step_cards.items():
            if key[0] == idx:
                card.set_status("done")
                card.collapse()

    def _ui_decompose(self, goals: list[str]) -> None:
        self._status.configure(text=f"> planned {len(goals)} goals", text_color=FG)
        if self._plan_edit_var.get():
            self._goal_lbl.configure(text="")
            self._populate_goal_boxes(goals)
            self._refine_btn.configure(state="normal", text="[ refine ]")
            self._refine_entry.delete(0, "end")
            self._plan_editor.pack(fill="x", pady=(2, 2))
        else:
            header = f"  plan  \u00b7  {len(goals)} goals"
            lines = "\n".join(f"  {i}. {trunc(g, 90)}" for i, g in enumerate(goals, 1))
            self._goal_lbl.configure(text=f"{header}\n{lines}")

    def _ui_done(self, result: str) -> None:
        self._status.configure(text="> done", text_color=FG_GREEN)
        self._goal_lbl.configure(text="")
        if self._step_cards:
            last_key = max(self._step_cards)
            self._step_cards[last_key].set_status("done")
        if self._result_card is not None:
            self._result_card.destroy()
        self._result_card = ResultCard(self._steps_frame, result)
        self._result_card.pack(fill="x", pady=(2, 0))
        self._scroll_steps_down()

    def _ui_error(self, msg: str) -> None:
        self._status.configure(text=f"> err: {trunc(msg, 50)}", text_color=FG_RED)

    def _ui_finished(self) -> None:
        self._running = False
        self._plan_editor.pack_forget()
        self._run_btn.configure(state="normal")
        self._pause_btn.configure(state="disabled", text="[ pause ]")
        self._stop_btn.configure(state="disabled")
        self._task_entry.configure(state="normal", text_color=FG)
        status_text = self._status.cget("text")
        if status_text.startswith("> stopping"):
            self._status.configure(text="> stopped", text_color=FG_RED)
        elif not status_text.startswith("> err") and not status_text.startswith("> done"):
            self._status.configure(text="> idle", text_color=FG_DIM)
        self.deiconify()
        self.lift()

    def _reopen(self) -> None:
        self.deiconify()
        self.lift()

    def _on_close(self) -> None:
        if self._callbacks:
            self._callbacks.stop_requested = True
            self._callbacks.pause_event.set()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "httpcore", "openai", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)
    MarkApp().mainloop()


if __name__ == "__main__":
    main()
