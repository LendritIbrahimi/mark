"""Action execution -- dispatches agent actions to MCP tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.state import StateManager

logger = logging.getLogger(__name__)

_ACTION_REGISTRY: list[tuple[str, str, str, str]] = [
    ("click",        "click",        "element_id",                     "Mouse"),
    ("double_click", "double_click", "element_id",                     "Mouse"),
    ("right_click",  "right_click",  "element_id",                     "Mouse"),
    ("drag",         "drag_to",      "from_element_id, to_element_id", "Mouse"),
    ("scroll",       "scroll_at",    "direction, amount",              "Mouse"),
    ("type_text",    "type_text",    "text, element_id?, submit?",     "Keyboard"),
    ("press_key",    "press_key",    "key",                            "Keyboard"),
    ("hotkey",       "hotkey_press", "keys",                           "Keyboard"),
    ("wait",         "wait",         "seconds",                        "Control"),
    ("done",         "done",         "text",                           "Control"),
]

_TOOL_MAP: dict[str, str] = {
    name: mcp for name, mcp, _, _ in _ACTION_REGISTRY
}
_VALID_PARAMS: dict[str, set[str]] = {
    name: {p.strip().rstrip("?") for p in params.split(",")}
    for name, _, params, _ in _ACTION_REGISTRY
}
_LOCAL_ACTION_DOCS: dict[str, str] = {
    "wait": "Pause 0.5-3s. Use liberally: after navigation, page loads, typing into search/URL bars, opening apps, or any time the UI needs to catch up",
    "done": "Mark task complete with a result description",
}


def build_action_docs(action_mcp: MCPClient) -> str:
    """Generate human-readable action documentation from MCP schemas."""
    schemas = action_mcp.tool_schemas
    groups: dict[str, list[str]] = {}

    for agent_name, mcp_name, params, group in _ACTION_REGISTRY:
        desc = (
            schemas.get(mcp_name, {}).get("description", "")
            or _LOCAL_ACTION_DOCS.get(agent_name, "")
        )
        line = f"- {agent_name}({params}) -- {desc}"
        groups.setdefault(group, []).append(line)

    parts: list[str] = []
    for group_name in ("Mouse", "Keyboard", "Control"):
        if group_name in groups:
            parts.append(f"{group_name}:")
            parts.extend(groups[group_name])
            parts.append("")

    return "\n".join(parts).rstrip()


def format_params(params: dict[str, Any]) -> str:
    """Human-readable params, e.g. wait(1s) instead of wait(seconds=1)."""
    if not params:
        return ""
    parts = []
    for v in params.values():
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


class ActionExecutor:
    """Dispatches agent actions to MCP tools with parameter resolution."""

    def __init__(
        self,
        action_mcp: MCPClient,
        state: StateManager,
        mcp_timeout: float = 30.0,
    ) -> None:
        self._action = action_mcp
        self._state = state
        self._timeout = mcp_timeout

    async def execute(self, name: str, params: dict) -> dict:
        """Execute a single action by name and return a result dict."""
        if name == "wait":
            seconds = max(0.5, min(float(params.get("seconds", 1)), 10.0))
            await asyncio.sleep(seconds)
            return {"success": True, "message": f"Waited {seconds:.1f}s"}

        if name == "done":
            text = params.get("text", "Task complete")
            return {"success": True, "message": text, "is_done": True}

        valid = _VALID_PARAMS.get(name)
        if valid is not None:
            params = {k: v for k, v in params.items() if k in valid}

        resolved = self._resolve_params(params)

        tool_name = _TOOL_MAP.get(name)
        if tool_name is None:
            return {"success": False, "message": f"Unknown action: {name}"}

        raw = await self._action.call_tool(tool_name, resolved, timeout=self._timeout)
        if isinstance(raw, dict):
            return raw
        return {"success": True, "message": str(raw)}

    def _resolve_params(self, params: dict) -> dict:
        """Replace element_id references with (x, y) coordinates."""
        resolved = dict(params)
        for key, x_key, y_key in (
            ("element_id", "x", "y"),
            ("from_element_id", "from_x", "from_y"),
            ("to_element_id", "to_x", "to_y"),
        ):
            if key in resolved:
                eid = resolved.pop(key)
                coords = self._state.resolve_element(eid)
                if coords is None:
                    raise ValueError(f"Element {eid} not found")
                resolved[x_key] = coords[0]
                resolved[y_key] = coords[1]
        return resolved
