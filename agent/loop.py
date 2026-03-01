"""Agent loop -- the main perceive-think-act cycle.

Connects to Vision and Action MCP servers, calls GPT-4o-mini for decisions,
and executes actions until the task is complete or limits are reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.debug import (
    create_session_dir,
    create_step_dir,
    remove_file_logger,
    save_b64_image,
    setup_file_logger,
    write_step_trace,
)
from agent.mcp_client import MCPClient
from agent.state import StateManager
from config import MarkConfig
from llm import create_llm_client
from llm.prompts import build_step_message, build_system_prompt

logger = logging.getLogger(__name__)


# -- Pydantic models for LLM structured output --


class ActionCall(BaseModel):
    name: str = Field(description="Action to execute")
    params: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    thought: str = Field(description="Reasoning about what to do next")
    actions: list[ActionCall] = Field(min_length=1)


# -- Agent-level actions (not routed to MCP) --

AGENT_ACTIONS = {"wait", "done"}


def _format_params(params: dict[str, Any]) -> str:
    """Human-readable action params: wait(1s, "reason") instead of wait(seconds=1, reason='...')."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'"{v}"')
        elif isinstance(v, list):
            parts.append("+".join(str(i) for i in v))
        elif isinstance(v, bool):
            parts.append(str(v).lower())
        elif isinstance(v, (int, float)):
            parts.append(str(v))
        else:
            parts.append(repr(v))
    return ", ".join(parts)


class AgentLoop:
    """Autonomous macOS desktop automation agent."""

    def __init__(
        self,
        task: str,
        config: MarkConfig,
        vision: MCPClient,
        action: MCPClient,
    ) -> None:
        self.task = task
        self.config = config
        self.vision = vision
        self.action = action
        self.llm = create_llm_client(config)
        self.state = StateManager(
            goal=task,
            max_steps=config.max_steps,
            max_recent_results=config.max_recent_results,
        )

        self._session_dir = create_session_dir(task)
        self._log_handler = setup_file_logger(self._session_dir)

        self._done = False
        self._consecutive_failures = 0
        self._screen_logged = False
        self._system_message = build_system_prompt()
        self._history: list[dict] = [self._system_message]

    async def run(self) -> str:
        """Run the perceive-think-act loop until done or limits reached."""
        logger.debug("Task: %s", self.task)
        if self.config.initial_delay:
            logger.debug("Waiting %.1fs for application to load...", self.config.initial_delay)
            await asyncio.sleep(self.config.initial_delay)

        for _ in range(self.config.max_steps):
            if self._done:
                break
            if self._consecutive_failures >= self.config.max_failures:
                logger.warning("Too many consecutive failures (%d), stopping.", self._consecutive_failures)
                break
            await self._step()

        if self._done:
            logger.info("Done.")
        else:
            logger.warning("Did not complete within %d steps.", self.config.max_steps)

        remove_file_logger(self._log_handler)
        return self.state.recent_results[-1] if self.state.recent_results else "No result."

    async def _step(self) -> None:
        """Execute one perceive -> think -> act cycle."""
        self.state.step += 1
        step = self.state.step
        await asyncio.sleep(self.config.step_delay)

        step_dir = create_step_dir(self._session_dir, step)
        timings: dict[str, float] = {}
        step_t0 = time.monotonic()

        try:
            # 1. PERCEIVE
            t0 = time.monotonic()
            perception = await self.vision.call_tool("observe", {
                "width": self.config.screenshot_width,
                "max_elements": self.config.max_elements,
                "use_omniparser": self.config.use_omniparser,
            }, timeout=self.config.mcp_timeout)
            timings["perceive_ms"] = (time.monotonic() - t0) * 1000
            self.state.update_ui(perception)
            logger.debug("Perceive: %.0fms (%d elements)", timings["perceive_ms"], len(self.state.element_positions))

            if not self._screen_logged:
                scale = perception.get("scale", 0)
                if scale:
                    real_w = int(self.config.screenshot_width * scale)
                    logger.info(
                        "Provider: %s | Model: %s | Max steps: %d | Screen: %dpx -> %dpx (scale %.2f)",
                        self.config.provider, self.llm.model, self.config.max_steps,
                        real_w, self.config.screenshot_width, scale,
                    )
                self._screen_logged = True

            self._save_screenshots(step_dir, perception)

            # 2. THINK
            t0 = time.monotonic()
            response = await self._think()
            timings["think_ms"] = (time.monotonic() - t0) * 1000
            logger.debug("Think: %.0fms", timings["think_ms"])

            self._save_llm_debug(step_dir, response)

            if response is None:
                self._write_trace(step_dir, None, [], timings, step_t0)
                return

            self._log_response(response, elapsed_s=timings["think_ms"] / 1000)

            # 3. ACT
            t0 = time.monotonic()
            results = await self._act(response)
            timings["act_ms"] = (time.monotonic() - t0) * 1000
            logger.debug("Act: %.0fms (%d actions)", timings["act_ms"], len(results))

            await asyncio.sleep(self.config.post_action_delay)

            self._consecutive_failures = 0
            self._write_trace(step_dir, response, results, timings, step_t0)

        except Exception as exc:
            self._consecutive_failures += 1
            logger.error("Step %d failed: %s", step, exc)
            self.state.record_result(f"Step error: {exc}")

    async def _think(self) -> StepResponse | None:
        """Build messages and call the LLM."""
        self._last_think_error: str | None = None
        step_msg = build_step_message(self.state)

        self._history.append(step_msg)
        self._trim_history()

        try:
            response = await self.llm.decide(
                self._history,
                StepResponse,
                image_b64=self.state.image_b64 if self.config.send_images else None,
            )
            response.actions = response.actions[:self.config.max_actions_per_step]
            self._history.append({
                "role": "assistant",
                "content": response.model_dump_json(),
            })
            return response

        except Exception as exc:
            self._history.pop()  # remove the user message to avoid consecutive user turns
            self._last_think_error = f"{type(exc).__name__}: {exc}"
            logger.error("LLM call failed: %s", self._last_think_error)
            self._consecutive_failures += 1
            self.state.record_result(f"LLM error: {exc}")
            return None

    async def _act(self, response: StepResponse) -> list[dict]:
        """Execute each action from the LLM response."""
        results: list[dict] = []

        for action in response.actions:
            name = action.name
            params = action.params
            action_key = f"{name}({sorted(params.items()) if params else ''})"

            try:
                result = await self._execute_action(name, params)
                results.append(result)

                if result.get("success"):
                    self.state.record_result(result.get("message", f"{name} ok"))
                    self.state.track_action(action_key, failed=False)
                else:
                    msg = result.get("message", f"{name} failed")
                    self.state.record_result(f"FAILED: {msg}")
                    self.state.track_action(action_key, failed=True)
                    break

                if result.get("is_done"):
                    self._done = True
                    break

            except Exception as exc:
                msg = f"{name} error: {exc}"
                logger.error(msg)
                self.state.record_result(f"FAILED: {msg}")
                self.state.track_action(action_key, failed=True)
                results.append({"success": False, "message": msg})
                break

        return results

    async def _execute_action(self, name: str, params: dict) -> dict:
        """Route an action to the appropriate handler."""
        # Agent-level actions
        if name == "wait":
            seconds = max(0.5, min(float(params.get("seconds", 1)), 10.0))
            await asyncio.sleep(seconds)
            return {"success": True, "message": f"Waited {seconds:.1f}s"}

        if name == "done":
            text = params.get("text", "Task complete")
            return {"success": True, "message": text, "is_done": True}

        # Resolve element_id to coordinates for MCP actions
        resolved_params = self._resolve_params(name, params)

        # Map agent action names to MCP tool names
        tool_map = {
            "click": "click",
            "double_click": "double_click",
            "right_click": "right_click",
            "hover": "hover_at",
            "drag": "drag_to",
            "scroll": "scroll_at",
            "type_text": "type_text",
            "press_key": "press_key",
            "hotkey": "hotkey_press",
        }

        tool_name = tool_map.get(name)
        if tool_name is None:
            return {"success": False, "message": f"Unknown action: {name}"}

        raw = await self.action.call_tool(
            tool_name, resolved_params, timeout=self.config.mcp_timeout,
        )
        if isinstance(raw, dict):
            return raw
        return {"success": True, "message": str(raw)}

    def _resolve_params(self, action_name: str, params: dict) -> dict:
        """Resolve element_id references to point-space coordinates."""
        resolved = dict(params)

        if "element_id" in resolved:
            eid = resolved.pop("element_id")
            coords = self.state.resolve_element(eid)
            if coords is None:
                raise ValueError(f"Element {eid} not found (total={len(self.state.element_positions)})")
            resolved["x"] = coords[0]
            resolved["y"] = coords[1]

        if "from_element_id" in resolved:
            eid = resolved.pop("from_element_id")
            coords = self.state.resolve_element(eid)
            if coords is None:
                raise ValueError(f"From-element {eid} not found")
            resolved["from_x"] = coords[0]
            resolved["from_y"] = coords[1]

        if "to_element_id" in resolved:
            eid = resolved.pop("to_element_id")
            coords = self.state.resolve_element(eid)
            if coords is None:
                raise ValueError(f"To-element {eid} not found")
            resolved["to_x"] = coords[0]
            resolved["to_y"] = coords[1]

        return resolved

    # -- Helpers --

    def _trim_history(self) -> None:
        """Keep conversation history within limits (preserve system message)."""
        max_msg = self.config.max_messages
        if len(self._history) <= max_msg + 1:
            return
        overflow = len(self._history) - max_msg - 1
        drop = overflow if overflow % 2 == 0 else overflow + 1
        self._history = [self._history[0]] + self._history[1 + drop:]
        logger.debug("Trimmed %d messages from history", drop)

    def _save_screenshots(self, step_dir: str, perception: dict) -> None:
        """Save raw and labeled screenshots to the step directory."""
        if perception.get("image"):
            save_b64_image(perception["image"], os.path.join(step_dir, "screenshot_labeled.jpg"))

    def _save_llm_debug(self, step_dir: str, response: StepResponse | None) -> None:
        """Save full LLM inputs/outputs for step-level debugging.

        Files written:
          elements.txt       -- element list the LLM saw
          llm_messages.json  -- full conversation history (text only)
          llm_response.json  -- structured LLM decision
        """
        with open(os.path.join(step_dir, "elements.txt"), "w", encoding="utf-8") as f:
            f.write(self.state.elements or "No elements detected")

        messages_log = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in self._history
        ]
        with open(os.path.join(step_dir, "llm_messages.json"), "w", encoding="utf-8") as f:
            json.dump({
                "step": self.state.step,
                "model": self.llm.model,
                "image_attached": bool(self.state.image_b64),
                "message_count": len(messages_log),
                "messages": messages_log,
            }, f, indent=2, default=str)

        raw_path = os.path.join(step_dir, "llm_raw_response.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(self.llm.last_raw_response or "(no response captured)")

        response_path = os.path.join(step_dir, "llm_response.json")
        if response:
            payload = response.model_dump()
        else:
            payload = {
                "error": "LLM call failed",
                "reason": getattr(self, "_last_think_error", None) or "unknown",
                "raw_response": self.llm.last_raw_response or None,
            }
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_trace(
        self,
        step_dir: str,
        response: StepResponse | None,
        results: list[dict],
        timings: dict[str, float],
        step_t0: float,
    ) -> None:
        timings["step_total_ms"] = (time.monotonic() - step_t0) * 1000
        trace = {
            "step": self.state.step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "thought": response.thought if response else "LLM failed",
            "actions": [{"name": a.name, "params": a.params} for a in response.actions] if response else [],
            "results": results,
            "element_count": len(self.state.element_positions),
            "loop_warning": self.state.loop_warning,
            "timings_ms": timings,
        }
        write_step_trace(step_dir, trace)

    def _log_response(self, response: StepResponse, elapsed_s: float = 0) -> None:
        step = self.state.step
        header = f"[Step {step} \u2014 {elapsed_s:.1f}s]" if elapsed_s else f"[Step {step}]"
        logger.info("%s %s", header, response.thought)
        for a in response.actions:
            logger.info("  \u2192 %s(%s)", a.name, _format_params(a.params))
