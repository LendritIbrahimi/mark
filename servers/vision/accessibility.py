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
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

logger = logging.getLogger(__name__)

INTERACTABLE_ROLES = {
    "AXButton", "AXTextField", "AXTextArea", "AXLink",
    "AXCheckBox", "AXRadioButton", "AXPopUpButton",
    "AXComboBox", "AXSlider", "AXMenuButton", "AXMenuItem",
    "AXTab", "AXImage", "AXStaticText", "AXIncrementor",
    "AXHeading", "AXDockItem", "AXMenuBarItem",
    "AXSheet", "AXDialog",
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
    "AXSheet": "Dialog",
    "AXDialog": "Dialog",
}

_DIALOG_ROLES = {"AXSheet", "AXDialog"}

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


def _collect_states(element: Any, role: str) -> list[str]:
    """Read boolean AX attributes to build a compact state-tag list."""
    states: list[str] = []
    if _ax_attr(element, "AXFocused"):
        states.append("FOCUSED")
    if _ax_attr(element, "AXSelected"):
        states.append("SELECTED")
    enabled = _ax_attr(element, "AXEnabled")
    if enabled is not None and not enabled:
        states.append("DISABLED")
    if role in ("AXCheckBox", "AXRadioButton") and _ax_attr(element, "AXValue"):
        states.append("CHECKED")
    if _ax_attr(element, "AXExpanded"):
        states.append("EXPANDED")
    return states


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
                "states": _collect_states(element, role),
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
    states: list[str]
    source: str = ""

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
        """e.g. [3] [FOCUSED] (Safari) Button: \"Submit\" """
        tags = " ".join(f"[{s}]" for s in self.states)
        parts = [f"[{self.id}]"]
        if tags:
            parts.append(tags)
        if self.source:
            parts.append(f"({self.source})")
        parts.append(f'{self.display_role}: "{self.label}"' if self.label else self.display_role)
        return " ".join(parts)


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


def _find_context_menu(app_ref: Any) -> Any | None:
    """If a context/popup menu is open, return its root AXMenu element.

    Follows the app's AXFocusedUIElement up through parents.  If the
    chain passes through an AXMenu *without* going through AXMenuBar,
    that menu is a context menu (not a menu-bar dropdown).
    """
    focused = _ax_attr(app_ref, "AXFocusedUIElement")
    if focused is None:
        return None

    menu = None
    current = focused
    for _ in range(20):
        role = _ax_attr(current, "AXRole") or ""
        if role == "AXMenu":
            menu = current
        elif role == "AXMenuBar":
            return None
        elif role == "AXApplication":
            break
        parent = _ax_attr(current, "AXParent")
        if parent is None:
            break
        current = parent
    return menu


def _get_dock_pid() -> int | None:
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.localizedName() == "Dock":
            return app.processIdentifier()
    return None


# ---------------------------------------------------------------------------
# Multi-app discovery via CGWindowList
# ---------------------------------------------------------------------------

_MIN_WINDOW_AREA = 50 * 50


def _get_onscreen_windows() -> list[dict]:
    """Return all normal-layer on-screen windows, ordered front-to-back.

    Each dict has keys: pid, x, y, w, h, owner.
    """
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID,
    )
    if not windows:
        return []

    result: list[dict] = []
    for w in windows:
        layer = int(w.get("kCGWindowLayer", 0))
        if layer != 0:
            continue

        bounds = w.get("kCGWindowBounds", {})
        x = int(bounds.get("X", 0))
        y = int(bounds.get("Y", 0))
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))

        if width * height < _MIN_WINDOW_AREA:
            continue

        result.append({
            "pid": int(w.get("kCGWindowOwnerPID", 0)),
            "x": x, "y": y, "w": width, "h": height,
            "owner": str(w.get("kCGWindowOwnerName", "?")),
        })
    return result


def _get_visible_app_pids(
    exclude_pids: set[int],
    windows: list[dict],
) -> list[tuple[int, str]]:
    """Unique (pid, app_name) pairs for visible background apps, front-to-back."""
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for w in windows:
        pid = w["pid"]
        if pid in exclude_pids or pid in seen:
            continue
        seen.add(pid)
        result.append((pid, w["owner"]))
    return result


def _is_occluded(cx: int, cy: int, elem_pid: int, windows: list[dict]) -> bool:
    """True if the point (cx, cy) is hidden behind a higher-z window from another app.

    Windows are ordered front-to-back.  If we encounter a window from the
    element's own PID first, the point is visible.  If we hit a foreign
    window that contains the point first, it's occluded.
    """
    for w in windows:
        if w["pid"] == elem_pid:
            return False
        if (w["x"] <= cx <= w["x"] + w["w"]
                and w["y"] <= cy <= w["y"] + w["h"]):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_accessibility_elements(
    max_elements: int = 150,
) -> tuple[list[UIElement], float]:
    """Read interactable elements from all visible apps and the Dock.

    Traverses AXChildren, AXMenuBar, and AXWindows of the frontmost app,
    then visible background apps (with a proportional budget), then the Dock.
    Background elements whose center is occluded by a higher-z window are
    discarded.  Off-screen elements and duplicates are also collapsed.

    Returns (elements, backing_scale) where backing_scale is the Retina
    factor needed by the labeler to map point-space onto the screenshot.
    """
    _DOCK_BUDGET = 30
    raw_limit = max_elements * 3

    # -- Frontmost app (gets first priority on the element budget) --

    app_raw: list[dict] = []

    pid, app_name = _get_frontmost_pid()
    if pid is not None:
        logger.debug("Frontmost app: %s (PID %d)", app_name, pid)
        app_ref = AXUIElementCreateApplication(pid)

        ctx_menu = _find_context_menu(app_ref)
        if ctx_menu is not None:
            _walk_tree(ctx_menu, app_raw, raw_limit, depth=1)

        focused_win = _ax_attr(app_ref, "AXFocusedWindow")
        if focused_win:
            for child in (_ax_attr(focused_win, "AXChildren") or []):
                if len(app_raw) >= raw_limit:
                    break
                child_role = _ax_attr(child, "AXRole") or ""
                if child_role in _DIALOG_ROLES:
                    _walk_tree(child, app_raw, raw_limit, depth=1)
            _walk_tree(focused_win, app_raw, raw_limit, depth=1)

        windows = _ax_attr(app_ref, "AXWindows")
        if windows:
            for win in windows:
                if len(app_raw) >= raw_limit:
                    break
                _walk_tree(win, app_raw, raw_limit, depth=1)

        menu_bar = _ax_attr(app_ref, "AXMenuBar")
        if menu_bar and len(app_raw) < raw_limit:
            _walk_tree(menu_bar, app_raw, raw_limit, depth=1)

        if len(app_raw) < raw_limit:
            _walk_tree(app_ref, app_raw, raw_limit)

        logger.debug("PID %d: %d raw elements collected", pid, len(app_raw))

    # -- Visible background apps (share the remaining budget) --

    onscreen_windows = _get_onscreen_windows()

    dock_pid = _get_dock_pid()
    exclude_pids: set[int] = set()
    if pid is not None:
        exclude_pids.add(pid)
    if dock_pid is not None:
        exclude_pids.add(dock_pid)

    bg_pids = _get_visible_app_pids(exclude_pids, onscreen_windows)

    if bg_pids:
        bg_budget = max(20, (raw_limit - len(app_raw)) // len(bg_pids))
        for bg_pid, bg_name in bg_pids:
            bg_raw: list[dict] = []
            bg_ref = AXUIElementCreateApplication(bg_pid)
            _walk_tree(bg_ref, bg_raw, bg_budget)
            for item in bg_raw:
                item["source"] = bg_name
                item["pid"] = bg_pid
            app_raw.extend(bg_raw)
            if bg_raw:
                logger.debug("BG app %s (PID %d): %d elements", bg_name, bg_pid, len(bg_raw))

    # -- Dock --

    dock_raw: list[dict] = []
    if dock_pid is not None:
        dock_ref = AXUIElementCreateApplication(dock_pid)
        _walk_tree(dock_ref, dock_raw, _DOCK_BUDGET)
        logger.debug("Dock: %d elements", len(dock_raw))

    raw = app_raw + dock_raw

    # -- Dedup, screen-clip, and occlusion filtering --

    backing_scale = _get_backing_scale()
    sw, sh = _get_screen_point_size()

    seen: set[tuple[int, int, int, int]] = set()
    visible: list[UIElement] = []
    skipped = dupes = occluded = 0

    for item in raw:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]

        if not _overlaps_screen(x, y, w, h, sw, sh):
            skipped += 1
            continue

        if item.get("source") and onscreen_windows:
            cx, cy = x + w // 2, y + h // 2
            if _is_occluded(cx, cy, item["pid"], onscreen_windows):
                occluded += 1
                continue

        key = (x, y, w, h)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)

        visible.append(UIElement(
            id=len(visible), role=item["role"], label=item["label"],
            x=x, y=y, w=w, h=h,
            states=item.get("states", []),
            source=item.get("source", ""),
        ))

    visible = visible[:max_elements]

    logger.debug(
        "Total: %d visible (%d off-screen, %d dupes, %d occluded, backing_scale=%.1f)",
        len(visible), skipped, dupes, occluded, backing_scale,
    )
    return visible, backing_scale
