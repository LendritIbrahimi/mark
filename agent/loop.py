"""Agent loop -- the main perceive-think-act cycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.callbacks import AgentCallbacks
from agent.config import MarkConfig
from agent.debug import (
    create_session_dir,
    create_step_dir,
    save_b64_image,
    write_step_trace,
)
from agent.executor import ActionExecutor, build_action_docs, format_params
from agent.llm import OpenAILLM
from agent.mcp_client import MCPClient
from agent.prompts import (
    build_goal_self_summary_messages,
    build_step_message,
    build_system_prompt,
)
from agent.state import StateManager

logger = logging.getLogger(__name__)


class ActionCall(BaseModel):
    """A single action the LLM wants to execute."""

    name: str = Field(description="Action to execute")
    params: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: str = Field(description="What you see on the screen right now")
    thought: str = Field(description="Reasoning about what to do next")
    actions: list[ActionCall] = Field(min_length=1)


class GoalSummary(BaseModel):
    """LLM-generated summary of a completed goal."""

    result: str


class AgentLoop:
    """Autonomous macOS desktop automation agent."""

    def __init__(
            self,
            task: str,
            config: MarkConfig,
            vision: MCPClient,
            action: MCPClient,
            callbacks: AgentCallbacks | None = None,
            goal_idx: int = 1,
            session_dir: str | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.vision = vision
        self.llm = OpenAILLM(config)
        self._cb = callbacks or AgentCallbacks()
        self.state = StateManager(
            goal=task,
            max_recent_results=config.max_recent_results,
        )
        self._executor = ActionExecutor(
            action, self.state, mcp_timeout=config.mcp_timeout,
        )

        self._goal_idx = goal_idx
        if config.save_debug_logs:
            self._session_dir = session_dir or create_session_dir(task)
        else:
            self._session_dir = ""
        self._done = False
        self._consecutive_failures = 0
        self._system_message = build_system_prompt(
            action_docs=build_action_docs(action),
        )
        self._history: list[dict] = [self._system_message]

    @property
    def session_dir(self) -> str:
        return self._session_dir

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> str:
        if self.config.initial_delay:
            await asyncio.sleep(self.config.initial_delay)

        while not self._done:
            if self._cb.stop_requested:
                logger.info("Stop requested by user.")
                break
            if self._consecutive_failures >= self.config.max_failures:
                logger.warning(
                    "Too many failures (%d), stopping.",
                    self._consecutive_failures,
                )
                break
            await self._step()
            if self.state.check_stale(self.config.max_stale_steps):
                logger.warning(
                    "Goal stuck for %d steps, aborting.",
                    self.config.max_stale_steps,
                )
                break

        if self._done:
            logger.info("Done.")

        result = await self._summarize_goal()
        logger.info("Goal result: %s", result)
        return result

    # ------------------------------------------------------------------
    # Step: perceive -> think -> act
    # ------------------------------------------------------------------

    async def _step(self) -> None:
        while not self._cb.pause_event.is_set():
            if self._cb.stop_requested:
                return
            await asyncio.sleep(0.1)

        self.state.step += 1
        step = self.state.step
        self._cb.emit("on_step_start", step)
        await asyncio.sleep(self.config.step_delay)

        if self.config.save_debug_logs:
            step_dir = create_step_dir(self._session_dir, step, self._goal_idx)
        else:
            step_dir = ""
        step_t0 = time.monotonic()

        try:
            await self._perceive(step_dir)

            if self._cb.get_guidance:
                guidance = self._cb.get_guidance()
                if guidance:
                    self.state.user_guidance = guidance

            response = await self._think()
            if self.config.save_debug_logs:
                self._save_step_debug(step_dir, response)

            if response is None:
                self._cb.emit(
                    "on_think", step,
                    "Failed to capture LLM response",
                    self._last_think_error or "Unknown LLM error",
                    [],
                )
                return

            self._log_response(response)
            self._cb.emit(
                "on_think", step,
                response.observation, response.thought,
                [{"name": a.name, "params": a.params} for a in response.actions],
            )

            results = await self._act(response)
            await asyncio.sleep(self.config.post_action_delay)

            self._consecutive_failures = 0
            if self.config.save_debug_logs:
                self._write_trace(step_dir, response, results, step_t0)

        except Exception as exc:
            self._consecutive_failures += 1
            logger.error("Step %d failed: %s", step, exc)
            self.state.record_result(f"Step error: {exc}")
            self._cb.emit("on_think", step, "Step failed", str(exc), [])

    async def _perceive(self, step_dir: str) -> None:
        perception = await self.vision.call_tool(
            "observe", {}, timeout=self.config.mcp_timeout,
        )
        self.state.update_ui(perception)

        if step_dir and perception.get("image"):
            save_b64_image(
                perception["image"],
                os.path.join(step_dir, "screenshot_labeled.jpg"),
            )

    async def _think(self) -> StepResponse | None:
        self._last_think_error: str | None = None
        step_msg = build_step_message(self.state)

        self._history.append(step_msg)
        self._trim_history()
        compressed = self._compress_history()

        try:
            response = await self.llm.decide(
                compressed,
                StepResponse,
                image_b64=self.state.image_b64 if self.config.send_images else None,
            )
            response.actions = response.actions[:self.config.max_actions_per_step]
            self._history.append({
                "role": "assistant",
                "content": response.model_dump_json(),
            })
            self.state.record_empty_response(success=True)
            return response

        except Exception as exc:
            self._history.pop()
            self._last_think_error = f"{type(exc).__name__}: {exc}"
            logger.error("LLM call failed: %s", self._last_think_error)
            self._consecutive_failures += 1
            self.state.record_empty_response(success=False)
            self.state.record_result(f"LLM error: {exc}")
            return None

    async def _act(self, response: StepResponse) -> list[dict]:
        results: list[dict] = []

        for i, action in enumerate(response.actions):
            if i > 0:
                await asyncio.sleep(0.3)

            name = action.name
            params = action.params
            action_key = f"{name}({sorted(params.items()) if params else ''})"

            elem_id = params.get("element_id")
            elem_label = ""
            if elem_id is not None:
                pos = self.state.element_positions.get(elem_id, {})
                elem_label = pos.get("label", "")
            label_part = f"('{elem_label}')" if elem_label else ""

            try:
                result = await self._executor.execute(name, params)
                results.append(result)

                if result.get("success"):
                    self.state.record_result(
                        f"{name}{label_part}: {result.get('message', f'{name} ok')}",
                    )
                    self.state.track_action(action_key, failed=False)
                else:
                    msg = result.get("message", f"{name} failed")
                    self.state.record_result(f"FAILED: {name}{label_part} → {msg}")
                    self.state.track_action(action_key, failed=True)
                    self._cb.emit("on_action_result", self.state.step, name, result)
                    break

                self._cb.emit("on_action_result", self.state.step, name, result)

                if result.get("is_done"):
                    self._done = True
                    self._cb.emit("on_done", result.get("message", "Done"))
                    break

            except Exception as exc:
                msg = f"{name} error: {exc}"
                logger.error(msg)
                self.state.record_result(f"FAILED: {msg}")
                self.state.track_action(action_key, failed=True)
                results.append({"success": False, "message": msg})
                break

        return results

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _compress_history(self) -> list[dict]:
        user_indices = [
            i for i, m in enumerate(self._history)
            if m.get("role") == "user"
        ]
        keep_full = set(user_indices[-2:])

        compressed: list[dict] = []
        for i, msg in enumerate(self._history):
            if msg.get("role") == "user" and i not in keep_full:
                compressed.append({
                    "role": "user",
                    "content": self._summarize_elements(msg["content"]),
                })
            else:
                compressed.append(msg)
        return compressed

    def _trim_history(self) -> None:
        max_msg = self.config.max_messages
        if len(self._history) <= max_msg + 1:
            return
        overflow = len(self._history) - max_msg - 1
        drop = overflow if overflow % 2 == 0 else overflow + 1
        self._history = [self._history[0]] + self._history[1 + drop:]

    @staticmethod
    def _summarize_elements(content: str) -> str:
        marker = "\nScreen elements:\n"
        idx = content.find(marker)
        if idx != -1:
            elem_text = content[idx + len(marker):]
            count = sum(1 for line in elem_text.splitlines() if line.strip())
            return content[:idx] + f"\nprevious elements: {count}"

        no_elem = "\nNo elements detected on screen."
        if no_elem in content:
            return content.split(no_elem)[0] + "\nprevious elements: 0"
        return content

    # ------------------------------------------------------------------
    # Goal summary
    # ------------------------------------------------------------------

    async def _summarize_goal(self) -> str:
        if not self.state.all_results:
            return "No actions were taken."
        messages = build_goal_self_summary_messages(
            self.task, list(self.state.all_results),
        )
        try:
            resp = await self.llm.decide(messages, GoalSummary)
            return resp.result
        except Exception as exc:
            logger.error("Goal summary failed: %s", exc)
            return self.state.all_results[-1]

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _save_step_debug(self, step_dir: str, response: StepResponse | None) -> None:
        with open(os.path.join(step_dir, "elements.txt"), "w", encoding="utf-8") as f:
            f.write(self.state.elements or "No elements detected")

        if response:
            payload = response.model_dump()
        else:
            payload = {
                "error": "LLM call failed",
                "reason": getattr(self, "_last_think_error", None) or "unknown",
                "raw_response": self.llm.last_raw_response or None,
            }
        with open(os.path.join(step_dir, "llm_response.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_trace(
        self,
        step_dir: str,
        response: StepResponse | None,
        results: list[dict],
        step_t0: float,
    ) -> None:
        elapsed_ms = (time.monotonic() - step_t0) * 1000
        trace = {
            "step": self.state.step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "observation": response.observation if response else "",
            "thought": response.thought if response else "LLM failed",
            "actions": [
                {"name": a.name, "params": a.params}
                for a in response.actions
            ] if response else [],
            "results": results,
            "element_count": len(self.state.element_positions),
            "loop_warning": self.state.loop_warning,
            "step_ms": elapsed_ms,
        }
        write_step_trace(step_dir, trace)

    def _log_response(self, response: StepResponse) -> None:
        step = self.state.step
        header = f"[Step {step}]"
        logger.info("%s [Obs] %s", header, response.observation)
        logger.info("%s [Think] %s", header, response.thought)
        for a in response.actions:
            logger.info("  \u2192 %s(%s)", a.name, format_params(a.params))
