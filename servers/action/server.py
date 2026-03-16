"""Action MCP Server -- mouse and keyboard actions."""

from __future__ import annotations

import json
import logging

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyProhibited,
)

NSApplication.sharedApplication().setActivationPolicy_(
    NSApplicationActivationPolicyProhibited,
)

logging.basicConfig(level=logging.WARNING)

from fastmcp import FastMCP

from servers.action import keyboard, mouse

mcp = FastMCP("mark-action")


def _result(ok: bool, message: str) -> str:
    return json.dumps(
        {"success": ok, "message": message},
    )


@mcp.tool()
def click(x: int, y: int) -> str:
    """Left-click on a UI element."""
    ok = mouse.click(x, y)
    msg = (
        f"Clicked at ({x}, {y})"
        if ok
        else f"Click at ({x}, {y}) failed"
    )
    return _result(ok, msg)


@mcp.tool()
def double_click(x: int, y: int) -> str:
    """Double-click on a UI element."""
    ok = mouse.double_click(x, y)
    msg = (
        f"Double-clicked at ({x}, {y})"
        if ok
        else f"Double-click at ({x}, {y}) failed"
    )
    return _result(ok, msg)


@mcp.tool()
def right_click(x: int, y: int) -> str:
    """Right-click to open context menu."""
    ok = mouse.right_click(x, y)
    msg = (
        f"Right-clicked at ({x}, {y})"
        if ok
        else f"Right-click at ({x}, {y}) failed"
    )
    return _result(ok, msg)


@mcp.tool()
def hover_at(x: int, y: int) -> str:
    """Hover over an element without clicking."""
    ok = mouse.hover(x, y)
    msg = (
        f"Hovered at ({x}, {y})"
        if ok
        else f"Hover at ({x}, {y}) failed"
    )
    return _result(ok, msg)


@mcp.tool()
def drag_to(
        from_x: int, from_y: int,
        to_x: int, to_y: int,
) -> str:
    """Drag from one element to another."""
    ok = mouse.drag(from_x, from_y, to_x, to_y)
    msg = (
        f"Dragged ({from_x},{from_y}) "
        f"-> ({to_x},{to_y})"
    )
    return _result(ok, msg if ok else f"Drag failed")


@mcp.tool()
def scroll_at(
        direction: str, amount: int = 3,
) -> str:
    """Scroll the page. direction: up/down/left/right."""
    ok = mouse.scroll(direction, amount)
    msg = (
        f"Scrolled {direction} by {amount}"
        if ok
        else f"Scroll {direction} failed"
    )
    return _result(ok, msg)


@mcp.tool()
def type_text(
        text: str,
        x: int | None = None,
        y: int | None = None,
        submit: bool = False,
) -> str:
    """Type text, optionally clicking a target first."""
    ok = keyboard.type_text(
        text, x=x, y=y, submit=submit,
    )
    where = f"at ({x}, {y})" if x is not None else "at cursor"
    msg = (
        f"Typed '{text}' {where}"
        if ok
        else f"Type failed {where}"
    )
    return _result(ok, msg)


@mcp.tool()
def press_key(key: str) -> str:
    """Press a single key: enter, tab, escape, etc."""
    ok = keyboard.press_key(key)
    msg = (
        f"Pressed {key}"
        if ok
        else f"Press {key} failed"
    )
    return _result(ok, msg)


@mcp.tool()
def hotkey_press(keys: list[str]) -> str:
    """Shortcut: ["command", "c"], etc."""
    ok = keyboard.hotkey(keys)
    combo = "+".join(keys)
    msg = (
        f"Pressed {combo}"
        if ok
        else f"Hotkey {combo} failed"
    )
    return _result(ok, msg)


if __name__ == "__main__":
    mcp.run()
