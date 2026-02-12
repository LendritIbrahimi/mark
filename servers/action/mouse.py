"""Mouse actions via pyautogui -- click, double-click, right-click, hover, drag, scroll."""

from __future__ import annotations

import logging
import time

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def click(x: int, y: int) -> bool:
    """Move to (x, y) and left-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        pyautogui.click()
        logger.debug("Left-clicked at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("click(%d, %d) failed: %s", x, y, exc)
        return False


def double_click(x: int, y: int) -> bool:
    """Move to (x, y) and double-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        pyautogui.doubleClick()
        logger.debug("Double-clicked at (%d, %d)", x, y)
        return True
    except Exception as exc:
        logger.error("double_click(%d, %d) failed: %s", x, y, exc)
        return False


def right_click(x: int, y: int) -> bool:
    """Move to (x, y) and right-click."""
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        pyautogui.rightClick()
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
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(to_x, to_y, duration=0.5)
        time.sleep(0.05)
        pyautogui.mouseUp()
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
        if direction in ("up", "down"):
            clicks = clamped * 3 if direction == "up" else -clamped * 3
            pyautogui.scroll(clicks)
        elif direction in ("left", "right"):
            clicks = -clamped * 3 if direction == "left" else clamped * 3
            pyautogui.hscroll(clicks)
        else:
            logger.error("scroll: invalid direction '%s'", direction)
            return False
        logger.debug("Scrolled %s by %d", direction, clamped)
        return True
    except Exception as exc:
        logger.error("scroll(%s, %d) failed: %s", direction, amount, exc)
        return False
