"""StepCard widget -- collapsible console-style step display."""

from __future__ import annotations

import customtkinter as ctk

from ui.theme import (
    BG, BORDER, FG, FG_DIM, FG_FAINT, HIGHLIGHT, MONO, PAD, SURFACE, WIDTH,
    trunc,
)


class StepCard(ctk.CTkFrame):
    _BADGES = {
        "running": ">>>",
        "done": " ok",
        "failed": "err",
    }

    def __init__(
        self, parent: ctk.CTkBaseClass, step_num: int,
        goal_idx: int = 0,
    ) -> None:
        super().__init__(
            parent, corner_radius=0,
            fg_color=SURFACE, border_width=1,
            border_color=BORDER,
        )
        self._expanded = True
        self._step_num = step_num

        tag = (
            f"{step_num:03d}"
            if goal_idx == 0
            else f"g{goal_idx}.{step_num:02d}"
        )

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=PAD, pady=(3, 0))

        self._title = ctk.CTkLabel(
            hdr, text=f"[{tag}] step",
            font=ctk.CTkFont(family=MONO, size=14),
            text_color=FG, anchor="w",
        )
        self._title.pack(side="left")

        self._badge = ctk.CTkLabel(
            hdr, text="[>>>]",
            font=ctk.CTkFont(family=MONO, size=14),
            text_color=FG_DIM, anchor="e",
        )
        self._badge.pack(side="right")

        for w in (hdr, self._title, self._badge):
            w.bind("<Button-1>", lambda _e: self._toggle())

        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(fill="x", padx=PAD, pady=(1, 3))

        mono = dict(
            font=ctk.CTkFont(family=MONO, size=13),
            anchor="w", justify="left",
            wraplength=WIDTH - 50,
        )
        self._obs = ctk.CTkLabel(self._body, text="", text_color=FG_FAINT, **mono)
        self._thought = ctk.CTkLabel(self._body, text="", text_color=FG_DIM, **mono)
        self._actions = ctk.CTkLabel(self._body, text="", text_color=FG, **mono)
        self._result = ctk.CTkLabel(self._body, text="", text_color=FG_DIM, **mono)

    def set_thinking(self, obs: str, thought: str, actions: list[dict]) -> None:
        self._obs.configure(text=f"  see: {trunc(obs, 80)}")
        self._obs.pack(fill="x")
        self._thought.configure(text=f"  think: {trunc(thought, 80)}")
        self._thought.pack(fill="x")
        parts = ", ".join(
            a["name"] + "(" + ", ".join(
                str(v) for v in a.get("params", {}).values()
            ) + ")"
            for a in actions
        )
        self._actions.configure(text=f"  act: {parts}")
        self._actions.pack(fill="x")

    def add_result(self, action_name: str, result: dict) -> None:
        ok = result.get("success", False)
        msg = result.get("message", "")
        tag = "ok" if ok else "FAIL"
        line = f"  --> {tag}: {trunc(msg, 70)}"
        prev = self._result.cget("text")
        self._result.configure(text=f"{prev}\n{line}" if prev else line)
        self._result.pack(fill="x")

    def set_status(self, status: str) -> None:
        badge = self._BADGES.get(status, "...")
        self._badge.configure(text=f"[{badge}]")

    def collapse(self) -> None:
        if self._expanded:
            self._toggle()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        if self._expanded:
            self._body.pack(fill="x", padx=PAD, pady=(1, 3))
        else:
            self._body.pack_forget()
