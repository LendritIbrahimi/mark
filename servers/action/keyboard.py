"""Keyboard actions via Quartz CGEvents with hardware-level event source.

Typing uses clipboard (pbcopy + Cmd-V).  Key presses and hotkeys use
CGEventCreateKeyboardEvent with CGEventSourceStateCombinedSessionState
so they are indistinguishable from real hardware input.
"""

from __future__ import annotations

import logging
import subprocess
import time

import pyautogui
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    CGEventSourceCreate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskShift,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskControl,
    kCGHIDEventTap,
    kCGEventSourceStateCombinedSessionState,
)

from servers.action.mouse import click as mouse_click

logger = logging.getLogger(__name__)

_source = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState)

# Virtual key codes (Carbon / HIToolbox)
_KEY_CODE_MAP: dict[str, int] = {
    "a": 0x00, "b": 0x0B, "c": 0x08, "d": 0x02, "e": 0x0E,
    "f": 0x03, "g": 0x05, "h": 0x04, "i": 0x22, "j": 0x26,
    "k": 0x28, "l": 0x25, "m": 0x2E, "n": 0x2D, "o": 0x1F,
    "p": 0x23, "q": 0x0C, "r": 0x0F, "s": 0x01, "t": 0x11,
    "u": 0x20, "v": 0x09, "w": 0x0D, "x": 0x07, "y": 0x10,
    "z": 0x06,
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15,
    "5": 0x17, "6": 0x16, "7": 0x1A, "8": 0x1C, "9": 0x19,
    "return": 0x24, "enter": 0x24,
    "tab": 0x30, "space": 0x31,
    "delete": 0x33, "backspace": 0x33, "forwarddelete": 0x75,
    "escape": 0x35, "esc": 0x35,
    "up": 0x7E, "down": 0x7D, "left": 0x7B, "right": 0x7C,
    "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76,
    "f5": 0x60, "f6": 0x61, "f7": 0x62, "f8": 0x64,
    "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
    "-": 0x1B, "=": 0x18, "[": 0x21, "]": 0x1E,
    ";": 0x29, "'": 0x27, ",": 0x2B, ".": 0x2F,
    "/": 0x2C, "\\": 0x2A, "`": 0x32,
}

_MODIFIER_MAP: dict[str, int] = {
    "command": kCGEventFlagMaskCommand,
    "cmd": kCGEventFlagMaskCommand,
    "shift": kCGEventFlagMaskShift,
    "option": kCGEventFlagMaskAlternate,
    "alt": kCGEventFlagMaskAlternate,
    "control": kCGEventFlagMaskControl,
    "ctrl": kCGEventFlagMaskControl,
}


def _post_key(key_code: int, down: bool, flags: int = 0) -> None:
    event = CGEventCreateKeyboardEvent(_source, key_code, down)
    CGEventSetFlags(event, flags)
    CGEventPost(kCGHIDEventTap, event)


def type_text(text: str, x: int | None = None, y: int | None = None, submit: bool = False) -> bool:
    """Type text via clipboard paste.

    If x/y are provided, clicks there first to focus. Otherwise types at cursor.
    Uses Cmd+A then Cmd+V when clicking a position (to replace existing text),
    or just Cmd+V when typing at cursor (to append).
    """
    try:
        if x is not None and y is not None:
            mouse_click(x, y)
            time.sleep(0.3)
            hotkey(["command", "a"])
            time.sleep(0.1)

        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        hotkey(["command", "v"])
        time.sleep(0.2)

        if submit:
            time.sleep(0.1)
            press_key("return")

        where = f"at ({x}, {y})" if x is not None else "at cursor"
        logger.info("Typed '%s' %s%s", text, where, " + submit" if submit else "")
        return True
    except Exception as exc:
        logger.error("type_text failed: %s", exc)
        return False


def press_key(key: str) -> bool:
    """Press and release a single key."""
    code = _KEY_CODE_MAP.get(key.lower().strip())
    if code is None:
        logger.error("press_key: unknown key '%s'", key)
        return False
    try:
        _post_key(code, True)
        time.sleep(0.01)
        _post_key(code, False)
        logger.debug("Pressed key: %s", key)
        return True
    except Exception as exc:
        logger.error("press_key('%s') failed: %s", key, exc)
        return False


def hotkey(keys: list[str]) -> bool:
    """Send a keyboard shortcut via CGEvent.

    keys: e.g. ["command", "s"], ["command", "shift", "n"]
    """
    if not keys:
        logger.error("hotkey: empty key list")
        return False

    lower_keys = [k.lower().strip() for k in keys]

    modifier_flags = 0
    main_key: str | None = None

    for k in lower_keys:
        if k in _MODIFIER_MAP:
            modifier_flags |= _MODIFIER_MAP[k]
        else:
            main_key = k

    if main_key is None:
        main_key = lower_keys[-1]

    key_code = _KEY_CODE_MAP.get(main_key)
    if key_code is None:
        logger.error("hotkey: unknown key '%s'", main_key)
        return False

    try:
        _post_key(key_code, True, modifier_flags)
        time.sleep(0.01)
        _post_key(key_code, False, modifier_flags)
        logger.debug("Sent hotkey: %s", "+".join(lower_keys))
        return True
    except Exception as exc:
        logger.error("hotkey failed: %s", exc)
        return False
