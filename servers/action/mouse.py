"""Mouse actions via Quartz CGEvents with hardware-level event source.

Uses CGEventSourceStateCombinedSessionState so synthetic clicks are
processed identically to real hardware clicks.  Before each click,
explicitly activates the target application and raises the specific
window via Accessibility APIs -- CGEventPost alone does NOT trigger
macOS application switching.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pyautogui
from AppKit import NSRunningApplication
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXUIElementPerformAction,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
    kAXErrorSuccess,
)
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    CGEventSetIntegerValueField,
    CGEventSourceCreate,
    CGWindowListCopyWindowInfo,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGHIDEventTap,
    kCGEventSourceStateCombinedSessionState,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
    kCGScrollEventUnitLine,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

_source = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState)
_kCGMouseEventClickState = 1
_NSApplicationActivateIgnoringOtherApps = 1 << 1


# ---------------------------------------------------------------------------
# Window activation helpers
# ---------------------------------------------------------------------------


def _activate_window_at(x: int, y: int) -> None:
    """Activate the app owning the topmost window at (x, y) and AXRaise it.

    CGEventPost does not trigger macOS application switching, so we must
    do it ourselves before posting synthetic click events.
    """
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    if not windows:
        return

    for w in windows:
        if int(w.get("kCGWindowLayer", 0)) != 0:
            continue

        bounds = w.get("kCGWindowBounds", {})
        wx = int(bounds.get("X", 0))
        wy = int(bounds.get("Y", 0))
        ww = int(bounds.get("Width", 0))
        wh = int(bounds.get("Height", 0))

        if not (wx <= x <= wx + ww and wy <= y <= wy + wh):
            continue

        pid = int(w.get("kCGWindowOwnerPID", 0))
        if not pid:
            return

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app:
            app.activateWithOptions_(_NSApplicationActivateIgnoringOtherApps)

        _raise_ax_window(pid, wx, wy, ww, wh)
        return


def _raise_ax_window(pid: int, wx: int, wy: int, ww: int, wh: int) -> None:
    """Find the AXWindow matching the given CGWindow bounds and AXRaise it."""
    app_ref = AXUIElementCreateApplication(pid)
    err, ax_windows = AXUIElementCopyAttributeValue(app_ref, "AXWindows", None)
    if err != kAXErrorSuccess or not ax_windows:
        return

    for ax_win in ax_windows:
        err, pos_val = AXUIElementCopyAttributeValue(ax_win, "AXPosition", None)
        if err != kAXErrorSuccess or pos_val is None:
            continue
        err, size_val = AXUIElementCopyAttributeValue(ax_win, "AXSize", None)
        if err != kAXErrorSuccess or size_val is None:
            continue

        try:
            ok, point = AXValueGetValue(pos_val, kAXValueCGPointType, None)
            if not ok:
                continue
            ok, size = AXValueGetValue(size_val, kAXValueCGSizeType, None)
            if not ok:
                continue
        except Exception:
            continue

        if (abs(point.x - wx) < 5 and abs(point.y - wy) < 5
                and abs(size.width - ww) < 5 and abs(size.height - wh) < 5):
            AXUIElementPerformAction(ax_win, "AXRaise")
            return


# ---------------------------------------------------------------------------
# Low-level CGEvent posting
# ---------------------------------------------------------------------------


def _post_mouse(
    event_type: int,
    x: int,
    y: int,
    button: int = kCGMouseButtonLeft,
    click_count: int = 1,
) -> None:
    event = CGEventCreateMouseEvent(_source, event_type, (x, y), button)
    if click_count != 1:
        CGEventSetIntegerValueField(event, _kCGMouseEventClickState, click_count)
    CGEventPost(kCGHIDEventTap, event)


# ---------------------------------------------------------------------------
# Public mouse actions
# ---------------------------------------------------------------------------


def click(x: int, y: int) -> bool:
    """Move to (x, y), activate the target window, and left-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseDown, x, y)
        time.sleep(0.01)
        _post_mouse(kCGEventLeftMouseUp, x, y)
        logger.debug("Left-clicked at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("click(%d, %d) failed: %s", x, y, exc)
        return False


def double_click(x: int, y: int) -> bool:
    """Move to (x, y), activate the target window, and double-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseDown, x, y, click_count=1)
        time.sleep(0.01)
        _post_mouse(kCGEventLeftMouseUp, x, y, click_count=1)
        time.sleep(0.02)
        _post_mouse(kCGEventLeftMouseDown, x, y, click_count=2)
        time.sleep(0.01)
        _post_mouse(kCGEventLeftMouseUp, x, y, click_count=2)
        logger.debug("Double-clicked at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("double_click(%d, %d) failed: %s", x, y, exc)
        return False


def right_click(x: int, y: int) -> bool:
    """Move to (x, y), activate the target window, and right-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(kCGEventRightMouseDown, x, y, kCGMouseButtonRight)
        time.sleep(0.01)
        _post_mouse(kCGEventRightMouseUp, x, y, kCGMouseButtonRight)
        logger.debug("Right-clicked at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("right_click(%d, %d) failed: %s", x, y, exc)
        return False


def hover(x: int, y: int) -> bool:
    """Move mouse to (x, y) without clicking."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        logger.debug("Hovered at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("hover(%d, %d) failed: %s", x, y, exc)
        return False


def drag(from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
    """Drag from one position to another."""
    try:
        pyautogui.moveTo(from_x, from_y, duration=0.3)
        _activate_window_at(from_x, from_y)
        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseDown, from_x, from_y)
        time.sleep(0.1)

        steps = 20
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(from_x + (to_x - from_x) * t)
            cy = int(from_y + (to_y - from_y) * t)
            _post_mouse(kCGEventLeftMouseDragged, cx, cy)
            time.sleep(0.5 / steps)

        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseUp, to_x, to_y)
        logger.debug("Dragged (%d,%d) -> (%d,%d)", from_x, from_y, to_x, to_y)
        return True
    except Exception as exc:
        logger.error("drag failed: %s", exc)
        return False


def scroll(direction: str, amount: int = 3) -> bool:
    """Scroll at the current mouse position.

    direction: "up", "down", "left", "right"
    amount: 1-10 scroll clicks
    """
    try:
        clamped = max(1, min(amount, 10))
        if direction == "up":
            ev = CGEventCreateScrollWheelEvent(_source, kCGScrollEventUnitLine, 1, clamped * 3)
        elif direction == "down":
            ev = CGEventCreateScrollWheelEvent(_source, kCGScrollEventUnitLine, 1, -clamped * 3)
        elif direction == "left":
            ev = CGEventCreateScrollWheelEvent(_source, kCGScrollEventUnitLine, 2, 0, -clamped * 3)
        elif direction == "right":
            ev = CGEventCreateScrollWheelEvent(_source, kCGScrollEventUnitLine, 2, 0, clamped * 3)
        else:
            logger.error("scroll: invalid direction '%s'", direction)
            return False
        CGEventPost(kCGHIDEventTap, ev)
        logger.debug("Scrolled %s by %d", direction, clamped)
        return True
    except Exception as exc:
        logger.error("scroll(%s, %d) failed: %s", direction, amount, exc)
        return False
