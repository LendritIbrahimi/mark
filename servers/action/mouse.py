"""Mouse actions via Quartz CGEvents."""

from __future__ import annotations

import logging
import time

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
    CGEventCreate,
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventGetLocation,
    CGEventPost,
    CGEventSetIntegerValueField,
    CGEventSourceCreate,
    CGWindowListCopyWindowInfo,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGEventMouseMoved,
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

_source = CGEventSourceCreate(
    kCGEventSourceStateCombinedSessionState,
)
_kCGMouseEventClickState = 1
_NSApplicationActivateIgnoringOtherApps = 1 << 1
_MOVE_STEPS = 18  # ~60 fps over 0.3 s


def _move_to(
        x: int, y: int, duration: float = 0.3,
) -> None:
    """Smooth cursor move using native Quartz events."""
    loc = CGEventGetLocation(CGEventCreate(None))
    steps = max(int(duration * 60), _MOVE_STEPS)
    interval = duration / steps
    for i in range(1, steps + 1):
        t = i / steps
        ix = loc.x + (x - loc.x) * t
        iy = loc.y + (y - loc.y) * t
        ev = CGEventCreateMouseEvent(
            _source, kCGEventMouseMoved,
            (ix, iy), kCGMouseButtonLeft,
        )
        CGEventPost(kCGHIDEventTap, ev)
        time.sleep(interval)


def _activate_window_at(x: int, y: int) -> None:
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
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

        if not (
                wx <= x <= wx + ww
                and wy <= y <= wy + wh
        ):
            continue

        pid = int(w.get("kCGWindowOwnerPID", 0))
        if not pid:
            return

        app = (
            NSRunningApplication
            .runningApplicationWithProcessIdentifier_(
                pid,
            )
        )
        if app:
            app.activateWithOptions_(
                _NSApplicationActivateIgnoringOtherApps,
            )

        _raise_ax_window(pid, wx, wy, ww, wh)
        return


def _raise_ax_window(
        pid: int, wx: int, wy: int, ww: int, wh: int,
) -> None:
    app_ref = AXUIElementCreateApplication(pid)
    err, ax_windows = AXUIElementCopyAttributeValue(
        app_ref, "AXWindows", None,
    )
    if err != kAXErrorSuccess or not ax_windows:
        return

    for ax_win in ax_windows:
        err, pos_val = AXUIElementCopyAttributeValue(
            ax_win, "AXPosition", None,
        )
        if err != kAXErrorSuccess or pos_val is None:
            continue
        err, size_val = AXUIElementCopyAttributeValue(
            ax_win, "AXSize", None,
        )
        if err != kAXErrorSuccess or size_val is None:
            continue

        try:
            ok, point = AXValueGetValue(
                pos_val, kAXValueCGPointType, None,
            )
            if not ok:
                continue
            ok, size = AXValueGetValue(
                size_val, kAXValueCGSizeType, None,
            )
            if not ok:
                continue
        except Exception:
            continue

        if (
                abs(point.x - wx) < 5
                and abs(point.y - wy) < 5
                and abs(size.width - ww) < 5
                and abs(size.height - wh) < 5
        ):
            AXUIElementPerformAction(
                ax_win, "AXRaise",
            )
            return


def _post_mouse(
        event_type: int,
        x: int,
        y: int,
        button: int = kCGMouseButtonLeft,
        click_count: int = 1,
) -> None:
    event = CGEventCreateMouseEvent(
        _source, event_type, (x, y), button,
    )
    if click_count != 1:
        CGEventSetIntegerValueField(
            event, _kCGMouseEventClickState,
            click_count,
        )
    CGEventPost(kCGHIDEventTap, event)


def click(x: int, y: int) -> bool:
    try:
        _move_to(x, y)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseDown, x, y)
        time.sleep(0.01)
        _post_mouse(kCGEventLeftMouseUp, x, y)
        return True
    except Exception as exc:
        logger.error(
            "click(%d, %d) failed: %s", x, y, exc,
        )
        return False


def double_click(x: int, y: int) -> bool:
    try:
        _move_to(x, y)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(
            kCGEventLeftMouseDown, x, y,
            click_count=1,
        )
        time.sleep(0.01)
        _post_mouse(
            kCGEventLeftMouseUp, x, y, click_count=1,
        )
        time.sleep(0.02)
        _post_mouse(
            kCGEventLeftMouseDown, x, y,
            click_count=2,
        )
        time.sleep(0.01)
        _post_mouse(
            kCGEventLeftMouseUp, x, y, click_count=2,
        )
        return True
    except Exception as exc:
        logger.error(
            "double_click(%d, %d) failed: %s",
            x, y, exc,
        )
        return False


def right_click(x: int, y: int) -> bool:
    try:
        _move_to(x, y)
        _activate_window_at(x, y)
        time.sleep(0.05)
        _post_mouse(
            kCGEventRightMouseDown, x, y,
            kCGMouseButtonRight,
        )
        time.sleep(0.01)
        _post_mouse(
            kCGEventRightMouseUp, x, y,
            kCGMouseButtonRight,
        )
        return True
    except Exception as exc:
        logger.error(
            "right_click(%d, %d) failed: %s",
            x, y, exc,
        )
        return False


def hover(x: int, y: int) -> bool:
    try:
        _move_to(x, y)
        return True
    except Exception as exc:
        logger.error(
            "hover(%d, %d) failed: %s", x, y, exc,
        )
        return False


def drag(
        from_x: int, from_y: int,
        to_x: int, to_y: int,
) -> bool:
    try:
        _move_to(from_x, from_y)
        _activate_window_at(from_x, from_y)
        time.sleep(0.05)
        _post_mouse(
            kCGEventLeftMouseDown, from_x, from_y,
        )
        time.sleep(0.1)

        steps = 20
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(from_x + (to_x - from_x) * t)
            cy = int(from_y + (to_y - from_y) * t)
            _post_mouse(
                kCGEventLeftMouseDragged, cx, cy,
            )
            time.sleep(0.5 / steps)

        time.sleep(0.05)
        _post_mouse(kCGEventLeftMouseUp, to_x, to_y)
        return True
    except Exception as exc:
        logger.error("drag failed: %s", exc)
        return False


def scroll(direction: str, amount: int = 3) -> bool:
    try:
        clamped = max(1, min(amount, 10))
        unit = kCGScrollEventUnitLine
        if direction == "up":
            ev = CGEventCreateScrollWheelEvent(
                _source, unit, 1, -clamped * 3,
            )
        elif direction == "down":
            ev = CGEventCreateScrollWheelEvent(
                _source, unit, 1, clamped * 3,
            )
        elif direction == "left":
            ev = CGEventCreateScrollWheelEvent(
                _source, unit, 2, 0, -clamped * 3,
            )
        elif direction == "right":
            ev = CGEventCreateScrollWheelEvent(
                _source, unit, 2, 0, clamped * 3,
            )
        else:
            logger.error(
                "scroll: invalid direction '%s'",
                direction,
            )
            return False
        CGEventPost(kCGHIDEventTap, ev)
        return True
    except Exception as exc:
        logger.error("scroll failed: %s", exc)
        return False
