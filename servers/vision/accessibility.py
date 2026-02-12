"""macOS Accessibility tree traversal via pyobjc.

Reads the UI element hierarchy from the frontmost application, its menu bar,
its windows, and the Dock, returning a flat list of interactable elements
with bounding boxes in point-space coordinates (matching pyautogui's click
coordinate system).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementGetPid,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
    kAXErrorSuccess,
)
from AppKit import NSScreen, NSWorkspace

logger = logging.getLogger(__name__)

INTERACTABLE_ROLES = {
    "AXButton", "AXTextField", "AXTextArea", "AXLink",
    "AXCheckBox", "AXRadioButton", "AXPopUpButton",
    "AXComboBox", "AXSlider", "AXMenuButton", "AXMenuItem",
    "AXTab", "AXImage", "AXStaticText", "AXIncrementor",
    "AXHeading", "AXDockItem", "AXMenuBarItem",
}

ROLE_DISPLAY_NAMES: dict[str, str] = {
    "AXButton": "Button",
    "AXTextField": "TextBox",
    "AXTextArea": "TextArea",
    "AXLink": "Link",
    "AXCheckBox": "Checkbox",
    "AXRadioButton": "Radio",
    "AXPopUpButton": "Dropdown",
    "AXComboBox": "ComboBox",
    "AXSlider": "Slider",
    "AXMenuButton": "MenuButton",
    "AXMenuItem": "MenuItem",
    "AXMenuBarItem": "Menu",
    "AXTab": "Tab",
    "AXImage": "Image",
    "AXStaticText": "Text",
    "AXIncrementor": "Stepper",
    "AXHeading": "Heading",
    "AXDockItem": "DockItem",
}

# ---------------------------------------------------------------------------
# Screen helpers
# ---------------------------------------------------------------------------


def _get_backing_scale() -> float:
    """Return the Retina backing scale factor (2.0 on Retina, 1.0 otherwise)."""
    screen = NSScreen.mainScreen()
    return float(screen.backingScaleFactor()) if screen else 1.0


def _get_screen_point_size() -> tuple[int, int]:
    """Return the main screen size in point-space."""
    screen = NSScreen.mainScreen()
    if screen is None:
        return 1920, 1080
    frame = screen.frame()
    return int(frame.size.width), int(frame.size.height)


def _overlaps_screen(x: int, y: int, w: int, h: int, sw: int, sh: int) -> bool:
    """True if the element rectangle overlaps the visible screen area."""
    return x + w > 0 and y + h > 0 and x < sw and y < sh


# ---------------------------------------------------------------------------
# AX attribute helpers
# ---------------------------------------------------------------------------


def _ax_attr(element: Any, attr: str) -> Any:
    """Safely read a single AX attribute, returning None on failure."""
    err, value = AXUIElementCopyAttributeValue(element, attr, None)
    return value if err == kAXErrorSuccess else None


def _ax_position(element: Any) -> tuple[int, int] | None:
    """Extract point-space position (x, y) from an AX element."""
    pos_val = _ax_attr(element, "AXPosition")
    if pos_val is None:
        return None
    try:
        ok, point = AXValueGetValue(pos_val, kAXValueCGPointType, None)
        return (int(point.x), int(point.y)) if ok else None
    except Exception:
        return None


def _ax_size(element: Any) -> tuple[int, int] | None:
    """Extract point-space size (w, h) from an AX element."""
    size_val = _ax_attr(element, "AXSize")
    if size_val is None:
        return None
    try:
        ok, size = AXValueGetValue(size_val, kAXValueCGSizeType, None)
        return (int(size.width), int(size.height)) if ok else None
    except Exception:
        return None


def _build_label(element: Any) -> str:
    """Build a human-readable label from available AX attributes.

    Intentionally excludes AXRoleDescription -- it returns generic strings
    like "search text field" for hidden system elements, producing phantom labels.
    """
    for attr in ("AXTitle", "AXDescription", "AXValue"):
        val = _ax_attr(element, attr)
        if val is None:
            continue
        text = str(val).strip()
        if not text:
            continue
        try:
            float(text)
            continue
        except ValueError:
            pass
        return text[:80]
    return ""


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

_MIN_SIZE = 5


def _walk_tree(
    element: Any,
    results: list[dict],
    max_elements: int,
    depth: int = 0,
    max_depth: int = 15,
) -> None:
    """Recursively walk the AX tree collecting interactable elements."""
    if len(results) >= max_elements or depth > max_depth:
        return

    role = _ax_attr(element, "AXRole") or ""

    if role in INTERACTABLE_ROLES:
        pos = _ax_position(element)
        size = _ax_size(element)
        if pos and size and size[0] >= _MIN_SIZE and size[1] >= _MIN_SIZE:
            results.append({
                "role": str(role),
                "label": _build_label(element),
                "x": pos[0], "y": pos[1],
                "w": size[0], "h": size[1],
            })

    children = _ax_attr(element, "AXChildren")
    if children:
        for child in children:
            if len(results) >= max_elements:
                break
            _walk_tree(child, results, max_elements, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# UIElement dataclass
# ---------------------------------------------------------------------------


@dataclass
class UIElement:
    """A single interactable UI element with point-space coordinates."""

    id: int
    role: str
    label: str
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def display_role(self) -> str:
        return ROLE_DISPLAY_NAMES.get(self.role, self.role.replace("AX", ""))

    def to_dict(self) -> dict:
        """Full element data (used internally by the labeler)."""
        return {
            "id": self.id, "role": self.role, "label": self.label,
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
        }

    def to_position(self) -> dict:
        """Minimal {id, x, y} with center coordinates for action resolution."""
        cx, cy = self.center
        return {"id": self.id, "x": cx, "y": cy}

    def format_entry(self) -> str:
        """e.g. [3] Button: \"Submit\" """
        if self.label:
            return f'[{self.id}] {self.display_role}: "{self.label}"'
        return f"[{self.id}] {self.display_role}"


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _get_frontmost_pid() -> tuple[int, str] | tuple[None, str]:
    """Return (pid, app_name) of the focused application.

    Uses the Accessibility system-wide element instead of NSWorkspace,
    which can return stale data in background subprocesses that don't
    run a Cocoa event loop.
    """
    system_wide = AXUIElementCreateSystemWide()
    err, focused_app = AXUIElementCopyAttributeValue(
        system_wide, "AXFocusedApplication", None,
    )
    if err != kAXErrorSuccess or focused_app is None:
        return None, "unknown"

    err, pid = AXUIElementGetPid(focused_app, None)
    if err != kAXErrorSuccess:
        return None, "unknown"

    name = _ax_attr(focused_app, "AXTitle") or "unknown"
    return int(pid), str(name)


def _get_dock_pid() -> int | None:
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.localizedName() == "Dock":
            return app.processIdentifier()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_accessibility_elements(
    max_elements: int = 150,
) -> tuple[list[UIElement], float]:
    """Read interactable elements from the frontmost app and the Dock.

    Traverses AXChildren, AXMenuBar, and AXWindows of the frontmost app,
    then the Dock.  Coordinates stay in point-space (matching pyautogui's
    click coordinate system).  Off-screen elements are discarded and
    duplicates (same position and size) are collapsed.

    Returns (elements, backing_scale) where backing_scale is the Retina
    factor needed by the labeler to map point-space onto the screenshot.
    """
    _DOCK_BUDGET = 30
    raw_limit = max_elements * 3

    app_raw: list[dict] = []

    pid, app_name = _get_frontmost_pid()
    if pid is not None:
        logger.info("Frontmost app: %s (PID %d)", app_name, pid)
        app_ref = AXUIElementCreateApplication(pid)

        _walk_tree(app_ref, app_raw, raw_limit)
        count_children = len(app_raw)

        menu_bar = _ax_attr(app_ref, "AXMenuBar")
        if menu_bar and len(app_raw) < raw_limit:
            _walk_tree(menu_bar, app_raw, raw_limit, depth=1)
        count_menu = len(app_raw) - count_children

        before_win = len(app_raw)
        windows = _ax_attr(app_ref, "AXWindows")
        if windows:
            for win in windows:
                if len(app_raw) >= raw_limit:
                    break
                _walk_tree(win, app_raw, raw_limit, depth=1)
        count_win = len(app_raw) - before_win

        logger.info(
            "PID %d: %d children, %d menu-bar, %d window elements",
            pid, count_children, count_menu, count_win,
        )

    dock_raw: list[dict] = []
    dock_pid = _get_dock_pid()
    if dock_pid is not None:
        dock_ref = AXUIElementCreateApplication(dock_pid)
        _walk_tree(dock_ref, dock_raw, _DOCK_BUDGET)
        logger.info("Dock: %d elements", len(dock_raw))

    raw = app_raw + dock_raw

    backing_scale = _get_backing_scale()
    sw, sh = _get_screen_point_size()

    seen: set[tuple[int, int, int, int]] = set()
    visible: list[UIElement] = []
    skipped = dupes = 0

    for item in raw:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]

        if not _overlaps_screen(x, y, w, h, sw, sh):
            skipped += 1
            continue

        key = (x, y, w, h)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)

        visible.append(UIElement(
            id=len(visible), role=item["role"], label=item["label"],
            x=x, y=y, w=w, h=h,
        ))

    visible = visible[:max_elements]

    logger.info(
        "Total: %d visible (%d off-screen, %d dupes, backing_scale=%.1f)",
        len(visible), skipped, dupes, backing_scale,
    )
    return visible, backing_scale
