"""Action MCP Server -- mouse and keyboard actions."""

from __future__ import annotations

import json
import logging

from AppKit import NSApplication, NSApplicationActivationPolicyProhibited
NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyProhibited)

from fastmcp import FastMCP

from servers.action import mouse, keyboard

import os as _os

_log_level = getattr(logging, _os.environ.get("MARK_LOG_LEVEL", "WARNING"), logging.WARNING)
logging.basicConfig(level=_log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logging.getLogger("mcp").setLevel(max(_log_level, logging.WARNING))
logger = logging.getLogger(__name__)

mcp = FastMCP("mark-action")


def _result(ok: bool, message: str) -> str:
    return json.dumps({"success": ok, "message": message})


# -- Mouse actions --


@mcp.tool()
def click(x: int, y: int) -> str:
    """Left-click on a UI element."""
    ok = mouse.click(x, y)
    return _result(ok, f"Clicked at ({x}, {y})" if ok else f"Click at ({x}, {y}) failed")


@mcp.tool()
def double_click(x: int, y: int) -> str:
    """Double-click on a UI element."""
    ok = mouse.double_click(x, y)
    return _result(ok, f"Double-clicked at ({x}, {y})" if ok else f"Double-click at ({x}, {y}) failed")


@mcp.tool()
def right_click(x: int, y: int) -> str:
    """Right-click on a UI element to open context menu."""
    ok = mouse.right_click(x, y)
    return _result(ok, f"Right-clicked at ({x}, {y})" if ok else f"Right-click at ({x}, {y}) failed")


@mcp.tool()
def hover_at(x: int, y: int) -> str:
    """Hover over an element without clicking. For tooltips and hover menus."""
    ok = mouse.hover(x, y)
    return _result(ok, f"Hovered at ({x}, {y})" if ok else f"Hover at ({x}, {y}) failed")


@mcp.tool()
def drag_to(from_x: int, from_y: int, to_x: int, to_y: int) -> str:
    """Drag from one element to another. For moving files, resizing, etc."""
    ok = mouse.drag(from_x, from_y, to_x, to_y)
    msg = f"Dragged ({from_x},{from_y}) -> ({to_x},{to_y})"
    return _result(ok, msg if ok else f"Drag failed: {msg}")


@mcp.tool()
def scroll_at(direction: str, amount: int = 3) -> str:
    """Scroll the page. direction: up/down/left/right, amount: 1-10."""
    ok = mouse.scroll(direction, amount)
    return _result(ok, f"Scrolled {direction} by {amount}" if ok else f"Scroll {direction} failed")


# -- Keyboard actions --


@mcp.tool()
def type_text(text: str, x: int | None = None, y: int | None = None, submit: bool = False) -> str:
    """Type text, optionally clicking a target element first. Set submit=true to press Enter after."""
    ok = keyboard.type_text(text, x=x, y=y, submit=submit)
    where = f"at ({x}, {y})" if x is not None else "at cursor"
    return _result(ok, f"Typed '{text}' {where}" if ok else f"Type failed {where}")


@mcp.tool()
def press_key(key: str) -> str:
    """Press a single key: enter, tab, escape, space, delete, arrows, f1-f12, letters, numbers."""
    ok = keyboard.press_key(key)
    return _result(ok, f"Pressed {key}" if ok else f"Press {key} failed")


@mcp.tool()
def hotkey_press(keys: list[str]) -> str:
    """Shortcut: ["command", "c"], ["command", "v"], ["command", "shift", "s"], etc."""
    ok = keyboard.hotkey(keys)
    combo = "+".join(keys)
    return _result(ok, f"Pressed {combo}" if ok else f"Hotkey {combo} failed")


if __name__ == "__main__":
    mcp.run()
