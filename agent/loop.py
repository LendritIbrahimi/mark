"""Agent loop -- the main perceive-think-act cycle.

Connects to Vision and Action MCP servers, calls the configured LLM for decisions,
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
from agent.llm import OpenAILLM
from agent.prompts import build_goal_self_summary_messages, build_step_message, build_system_prompt
from config import MarkConfig

logger = logging.getLogger(__name__)


# -- Pydantic models for LLM structured output --


class ActionCall(BaseModel):
    name: str = Field(description="Action to execute")
    params: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    thought: str = Field(description="Reasoning about what to do next")
    actions: list[ActionCall] = Field(min_length=1)


class GoalSummary(BaseModel):
    result: str


# -- Action registry: single source of truth for agent ↔ MCP mapping --
# (agent_name, mcp_tool, agent_params, group)

_ACTION_REGISTRY: list[tuple[str, str, str, str]] = [
    ("click",        "click",        "element_id",                      "Mouse"),
    ("double_click", "double_click", "element_id",                      "Mouse"),
    ("right_click",  "right_click",  "element_id",                      "Mouse"),
    ("hover",        "hover_at",     "element_id",                      "Mouse"),
    ("drag",         "drag_to",      "from_element_id, to_element_id",  "Mouse"),
    ("scroll",       "scroll_at",    "direction, amount",               "Mouse"),
    ("type_text",    "type_text",    "text, element_id?, submit?",      "Keyboard"),
    ("press_key",    "press_key",    "key",                             "Keyboard"),
    ("hotkey",       "hotkey_press", "keys",                            "Keyboard"),
    ("wait",         "wait",         "seconds",                         "Control"),
    ("done",         "done",         "text",                            "Control"),
]

_TOOL_MAP: dict[str, str] = {name: mcp for name, mcp, _, _ in _ACTION_REGISTRY}
_VALID_PARAMS: dict[str, set[str]] = {
    name: {p.strip().rstrip("?") for p in params.split(",")}
    for name, _, params, _ in _ACTION_REGISTRY
}


_LOCAL_ACTION_DOCS: dict[str, str] = {
    "wait": "Pause 0.5-10 seconds for loading or when unsure what to do",
    "done": "Mark task complete with a result description",
}


def _build_action_docs(action_mcp: MCPClient) -> str:
    """Generate action documentation from MCP tool schemas + the agent's parameter mapping."""
    schemas = action_mcp.tool_schemas
    groups: dict[str, list[str]] = {}

    for agent_name, mcp_name, params, group in _ACTION_REGISTRY:
        desc = (schemas.get(mcp_name, {}).get("description", "")
                or _LOCAL_ACTION_DOCS.get(agent_name, ""))
        line = f"- {agent_name}({params}) -- {desc}"
        groups.setdefault(group, []).append(line)

    parts: list[str] = []
    for group_name in ("Mouse", "Keyboard", "Control"):
        if group_name in groups:
            parts.append(f"{group_name}:")
            parts.extend(groups[group_name])
            parts.append("")

    return "\n".join(parts).rstrip()


def _format_params(params: dict[str, Any]) -> str:
    """Human-readable action params: wait(1s) instead of wait(seconds=1)."""
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
        self.llm = OpenAILLM(config)
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
        self._system_message = build_system_prompt(
            action_docs=_build_action_docs(action),
        )
        self._history: list[dict] = [self._system_message]

    @property
    def session_dir(self) -> str:
        return self._session_dir

    async def run(self) -> str:
        """Run the perceive-think-act loop and return an LLM summary of the goal."""
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

        result = await self._summarize_goal()
        logger.info("Goal result: %s", result)

        remove_file_logger(self._log_handler)
        return result

    async def _summarize_goal(self) -> str:
        """Call the LLM to produce a self-contained summary of what this goal accomplished."""
        if not self.state.recent_results:
            return "No actions were taken."
        messages = build_goal_self_summary_messages(self.task, list(self.state.recent_results))
        try:
            response = await self.llm.decide(messages, GoalSummary)
            return response.result
        except Exception as exc:
            logger.warning("Goal summary LLM call failed: %s", exc)
            return self.state.recent_results[-1]

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
            }, timeout=self.config.mcp_timeout)
            timings["perceive_ms"] = (time.monotonic() - t0) * 1000
            self.state.update_ui(perception)
            logger.debug("Perceive: %.0fms (%d elements)", timings["perceive_ms"], len(self.state.element_positions))

            if not self._screen_logged:
                scale = perception.get("scale", 0)
                if scale:
                    real_w = int(self.config.screenshot_width * scale)
                    logger.info(
                        "Model: %s | Max steps: %d | Screen: %dpx -> %dpx (scale %.2f)",
                        self.llm.model, self.config.max_steps,
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
        if name == "wait":
            seconds = max(0.5, min(float(params.get("seconds", 1)), 10.0))
            await asyncio.sleep(seconds)
            return {"success": True, "message": f"Waited {seconds:.1f}s"}

        if name == "done":
            text = params.get("text", "Task complete")
            return {"success": True, "message": text, "is_done": True}

        valid = _VALID_PARAMS.get(name)
        if valid is not None:
            extra = set(params) - valid
            if extra:
                logger.debug("Stripping unexpected params from %s: %s", name, extra)
                params = {k: v for k, v in params.items() if k in valid}

        resolved_params = self._resolve_params(name, params)

        tool_name = _TOOL_MAP.get(name)
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

    @staticmethod
    def _summarize_elements(content: str) -> str:
        """Replace the element block in a user message with a short count."""
        marker = "\nScreen elements:\n"
        idx = content.find(marker)
        if idx != -1:
            elem_text = content[idx + len(marker):]
            count = sum(1 for line in elem_text.splitlines() if line.strip())
            return content[:idx] + f"\nprevious elements on the screen: {count}"
        if "\nNo elements detected on screen." in content:
            return content.split("\nNo elements detected on screen.")[0] + "\nprevious elements on the screen: 0"
        return content

    def _compress_history(self) -> list[dict]:
        """Return a copy of _history with element lists summarized except for the current user message."""
        user_indices = [i for i, m in enumerate(self._history) if m.get("role") == "user"]
        keep_full = set(user_indices[-1:])

        compressed: list[dict] = []
        for i, msg in enumerate(self._history):
            if msg.get("role") == "user" and i not in keep_full:
                compressed.append({"role": "user", "content": self._summarize_elements(msg["content"])})
            else:
                compressed.append(msg)
        return compressed

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
          elements.txt              -- element list the LLM saw
          llm_messages.json         -- full conversation history (text only)
          llm_response.json         -- structured LLM decision
          llm_raw_response.txt      -- raw LLM output string
          llm_response_metadata.json -- full LLM HTTP response metadata
        """
        with open(os.path.join(step_dir, "elements.txt"), "w", encoding="utf-8") as f:
            f.write(self.state.elements or "No elements detected")

        messages_log = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in self._compress_history()
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

        if self.llm.last_response_metadata:
            meta_path = os.path.join(step_dir, "llm_response_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.llm.last_response_metadata, f, indent=2, default=str)

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
        calls = f"({self.llm.successful_calls}/{self.llm.total_calls})"
        header = f"[Step {step} \u2014 {elapsed_s:.1f}s] {calls}" if elapsed_s else f"[Step {step}] {calls}"
        logger.info("%s %s", header, response.thought)
        for a in response.actions:
            logger.info("  \u2192 %s(%s)", a.name, _format_params(a.params))
