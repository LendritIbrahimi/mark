"""macOS Accessibility tree traversal via pyobjc."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from AppKit import NSScreen, NSWorkspace
from ApplicationServices import (
    AXUIElementCopyActionNames,
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementSetAttributeValue,
    AXUIElementGetPid,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
    kAXErrorSuccess,
)
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

logger = logging.getLogger(__name__)

_UNLABELED_ROLES = {
    "AXGroup", "AXScrollArea", "AXSplitGroup",
    "AXTabGroup", "AXWebArea", "AXWindow",
    "AXList", "AXOutline", "AXTable",
    "AXMenuBar", "AXToolbar",
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
    "AXIcon": "Icon",
    "AXSheet": "Dialog",
    "AXDialog": "Dialog",
    "AXDisclosureTriangle": "Toggle",
    "AXCell": "Cell",
    "AXGroup": "Group",
    "AXScrollArea": "ScrollArea",
    "AXList": "List",
    "AXToolbar": "Toolbar",
    "AXWindow": "Window",
    "AXOutline": "Outline",
    "AXTable": "Table",
    "AXRow": "Row",
    "AXColumn": "Column",
    "AXSplitGroup": "SplitGroup",
    "AXTabGroup": "TabGroup",
    "AXMenu": "Menu",
    "AXMenuBar": "MenuBar",
    "AXWebArea": "WebArea",
}

_DIALOG_ROLES = {"AXSheet", "AXDialog"}
_ACTION_REQUIRED_ROLES = {"AXStaticText", "AXImage"}
_CLICKABLE_ACTIONS = {
    "AXPress", "AXIncrement", "AXDecrement",
    "AXConfirm", "AXCancel", "AXRaise",
    "AXSetValue",
}
_TEXT_ROLES = {"AXTextField", "AXTextArea"}
_CONTAINER_ROLES = {"AXCell", "AXGroup"}
_SCROLLABLE_ROLES = {"AXScrollArea"}

_SUBROLE_DISPLAY: dict[str, str] = {
    "AXSearchField": "SearchField",
    "AXSecureTextField": "PasswordField",
    "AXCloseButton": "CloseButton",
    "AXMinimizeButton": "MinButton",
    "AXZoomButton": "ZoomButton",
    "AXFullScreenButton": "FullScreenButton",
    "AXToolbarButton": "ToolbarButton",
    "AXSortButton": "SortButton",
}

_GENERIC_ROLE_DESCRIPTIONS = {
                                 v.lower() for v in ROLE_DISPLAY_NAMES.values()
                             } | {
                                 "text", "button", "image", "link",
                                 "checkbox", "radio button",
                                 "pop up button", "combo box", "slider",
                                 "menu item", "menu bar item", "menu bar",
                                 "tab", "stepper", "heading",
                                 "text field", "text area",
                                 "group", "list", "table", "row", "column",
                                 "outline", "toolbar", "scroll area",
                                 "split group", "tab group", "radio group",
                                 "window", "web area", "icon",
                             }


def _get_backing_scale() -> float:
    screen = NSScreen.mainScreen()
    if screen:
        return float(screen.backingScaleFactor())
    return 1.0


def _get_screen_point_size() -> tuple[int, int]:
    screen = NSScreen.mainScreen()
    if screen is None:
        return 1920, 1080
    frame = screen.frame()
    return int(frame.size.width), int(frame.size.height)


def _overlaps_screen(
        x: int, y: int, w: int, h: int,
        sw: int, sh: int,
) -> bool:
    return (
            x + w > 0
            and y + h > 0
            and x < sw
            and y < sh
    )


def _ax_attr(element: Any, attr: str) -> Any:
    err, value = AXUIElementCopyAttributeValue(
        element, attr, None,
    )
    return value if err == kAXErrorSuccess else None


def _ax_actions(element: Any) -> list[str]:
    try:
        err, actions = AXUIElementCopyActionNames(
            element, None,
        )
        if err == kAXErrorSuccess and actions:
            return list(actions)
    except Exception:
        pass
    return []


def _ax_position(
        element: Any,
) -> tuple[int, int] | None:
    pos_val = _ax_attr(element, "AXPosition")
    if pos_val is None:
        return None
    try:
        ok, point = AXValueGetValue(
            pos_val, kAXValueCGPointType, None,
        )
        if ok:
            return (int(point.x), int(point.y))
        return None
    except Exception:
        return None


def _ax_size(
        element: Any,
) -> tuple[int, int] | None:
    size_val = _ax_attr(element, "AXSize")
    if size_val is None:
        return None
    try:
        ok, size = AXValueGetValue(
            size_val, kAXValueCGSizeType, None,
        )
        if ok:
            return (int(size.width), int(size.height))
        return None
    except Exception:
        return None


def _build_label(
        element: Any, role: str = "",
) -> str:
    for attr in ("AXTitle", "AXDescription", "AXValue"):
        val = _ax_attr(element, attr)
        if val is None or not isinstance(val, str):
            continue
        text = val.strip()
        if not text:
            continue
        if role not in _TEXT_ROLES:
            try:
                float(text)
                continue
            except ValueError:
                pass
        return text[:80]
    if role in _TEXT_ROLES:
        ph = _ax_attr(element, "AXPlaceholderValue")
        if isinstance(ph, str):
            text = ph.strip()
            if text:
                return text[:80]
    help_val = _ax_attr(element, "AXHelp")
    if isinstance(help_val, str):
        text = help_val.strip()
        if text:
            return text[:80]
    rd = _ax_attr(element, "AXRoleDescription")
    if isinstance(rd, str):
        text = rd.strip()
        if text.lower() not in _GENERIC_ROLE_DESCRIPTIONS:
            return text[:80]
    if role in _CONTAINER_ROLES:
        text = _label_from_children(element)
        if text:
            return text
    return ""


def _label_from_children(element: Any) -> str:
    children = _ax_attr(element, "AXChildren")
    if not children:
        return ""
    for child in children[:5]:
        cr = _ax_attr(child, "AXRole") or ""
        if cr == "AXStaticText":
            for attr in (
                    "AXValue", "AXTitle", "AXDescription",
            ):
                val = _ax_attr(child, attr)
                if isinstance(val, str):
                    text = val.strip()
                    if text:
                        return text[:80]
    return ""


def _collect_states(
        element: Any, role: str,
) -> list[str]:
    states: list[str] = []
    if _ax_attr(element, "AXFocused"):
        states.append("FOCUSED")
    if _ax_attr(element, "AXSelected"):
        states.append("SELECTED")
    enabled = _ax_attr(element, "AXEnabled")
    if enabled is not None and not enabled:
        states.append("DISABLED")
    if (
            role in ("AXCheckBox", "AXRadioButton")
            and _ax_attr(element, "AXValue")
    ):
        states.append("CHECKED")
    if _ax_attr(element, "AXExpanded"):
        states.append("EXPANDED")
    if _ax_attr(element, "AXRequired"):
        states.append("REQUIRED")
    if (
            role == "AXLink"
            and _ax_attr(element, "AXVisited")
    ):
        states.append("VISITED")
    return states


_MIN_SIZE = 5


def _intersect_rects(
        r1: tuple[int, int, int, int] | None,
        r2: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if r1 is None:
        return r2
    return (
        max(r1[0], r2[0]),
        max(r1[1], r2[1]),
        min(r1[2], r2[2]),
        min(r1[3], r2[3]),
    )


def _center_in_rect(
        cx: int, cy: int,
        rect: tuple[int, int, int, int] | None,
) -> bool:
    if rect is None:
        return True
    return (
            rect[0] <= cx <= rect[2]
            and rect[1] <= cy <= rect[3]
    )


def _walk_tree(
        element: Any,
        results: list[dict],
        max_elements: int,
        depth: int = 0,
        max_depth: int = 15,
        path: str = "",
        clip_rect: tuple[int, int, int, int] | None = None,
) -> None:
    if len(results) >= max_elements or depth > max_depth:
        return

    role = _ax_attr(element, "AXRole") or ""

    pos = _ax_position(element)
    size = _ax_size(element)

    child_clip = clip_rect
    if (
            role in _SCROLLABLE_ROLES
            and pos and size
            and size[0] >= _MIN_SIZE
            and size[1] >= _MIN_SIZE
    ):
        bounds = (
            pos[0], pos[1],
            pos[0] + size[0], pos[1] + size[1],
        )
        child_clip = _intersect_rects(
            clip_rect, bounds,
        )

    if (
            pos and size
            and size[0] >= _MIN_SIZE
            and size[1] >= _MIN_SIZE
    ):
        labeled = role not in _UNLABELED_ROLES
        if role in _ACTION_REQUIRED_ROLES and labeled:
            actions = _ax_actions(element)
            labeled = bool(
                _CLICKABLE_ACTIONS.intersection(actions),
            )
        elif not labeled:
            actions = _ax_actions(element)
            if _CLICKABLE_ACTIONS.intersection(actions):
                labeled = True

        cx = pos[0] + size[0] // 2
        cy = pos[1] + size[1] // 2
        offscreen = not _center_in_rect(
            cx, cy, clip_rect,
        )
        states = _collect_states(element, role)
        if offscreen:
            states.append("OFFSCREEN")
            labeled = False

        results.append({
            "role": str(role),
            "subrole": str(
                _ax_attr(element, "AXSubrole") or "",
            ),
            "label": _build_label(element, role=role),
            "x": pos[0],
            "y": pos[1],
            "w": size[0],
            "h": size[1],
            "states": states,
            "path": path,
            "depth": depth,
            "labeled": labeled,
        })

    children = _ax_attr(element, "AXChildren")
    if children:
        child_list = list(children)

        role_counts: dict[str, int] = {}
        child_roles: list[str] = []
        for child in child_list:
            cr = _ax_attr(child, "AXRole") or ""
            child_roles.append(cr)
            role_counts[cr] = (
                    role_counts.get(cr, 0) + 1
            )

        role_indices: dict[str, int] = {}
        for child, cr in zip(child_list, child_roles):
            if len(results) >= max_elements:
                break
            idx = role_indices.get(cr, 0)
            role_indices[cr] = idx + 1
            if role_counts[cr] > 1:
                component = f"{cr}[{idx}]"
            else:
                component = cr
            if path:
                child_path = f"{path}/{component}"
            else:
                child_path = component
            if cr == "AXWebArea":
                child_max_depth = depth + 25
            else:
                child_max_depth = max_depth
            _walk_tree(
                child, results, max_elements,
                depth + 1, child_max_depth,
                path=child_path,
                clip_rect=child_clip,
            )


@dataclass
class UIElement:
    """A single UI element from the accessibility tree."""

    id: int
    role: str
    label: str
    x: int
    y: int
    w: int
    h: int
    states: list[str]
    source: str = ""
    subrole: str = ""
    path: str = ""
    depth: int = 0
    labeled: bool = True

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def display_role(self) -> str:
        if self.subrole in _SUBROLE_DISPLAY:
            return _SUBROLE_DISPLAY[self.subrole]
        return ROLE_DISPLAY_NAMES.get(
            self.role, self.role.replace("AX", ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }

    def to_position(self) -> dict:
        cx, cy = self.center
        return {
            "id": self.id, "x": cx, "y": cy,
            "label": self.label, "role": self.role,
        }

    def format_entry(self) -> str:
        indent = "  " * self.depth
        parts = [f"[{self.id}]"]
        tags = " ".join(
            f"[{s}]" for s in self.states
        )
        if tags:
            parts.append(tags)
        if self.source:
            parts.append(f"({self.source})")
        if self.label:
            parts.append(
                f'{self.display_role}: "{self.label}"',
            )
        else:
            parts.append(self.display_role)
        return indent + " ".join(parts)


def _get_frontmost_pid() -> (
        tuple[int, str] | tuple[None, str]
):
    system_wide = AXUIElementCreateSystemWide()
    err, focused_app = AXUIElementCopyAttributeValue(
        system_wide, "AXFocusedApplication", None,
    )
    if err == kAXErrorSuccess and focused_app is not None:
        err, pid = AXUIElementGetPid(
            focused_app, None,
        )
        if err == kAXErrorSuccess:
            name = (
                    _ax_attr(focused_app, "AXTitle")
                    or "unknown"
            )
            return int(pid), str(name)

    ws = NSWorkspace.sharedWorkspace()
    active = ws.frontmostApplication()
    if active and active.processIdentifier():
        pid = int(active.processIdentifier())
        name = str(active.localizedName() or "unknown")
        return pid, name

    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
    if windows:
        for w in windows:
            layer = int(w.get("kCGWindowLayer", 0))
            if layer == 0:
                pid = int(
                    w.get("kCGWindowOwnerPID", 0),
                )
                name = str(
                    w.get("kCGWindowOwnerName", "unknown"),
                )
                return pid, name

    return None, "unknown"


def _ensure_accessibility(app_ref: Any) -> None:
    err = AXUIElementSetAttributeValue(
        app_ref, "AXManualAccessibility", True,
    )
    if err == kAXErrorSuccess:
        return
    AXUIElementSetAttributeValue(
        app_ref, "AXEnhancedUserInterface", True,
    )


def _find_context_menu(app_ref: Any) -> Any | None:
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
    ws = NSWorkspace.sharedWorkspace()
    for app in ws.runningApplications():
        if app.localizedName() == "Dock":
            return app.processIdentifier()
    return None


def _get_finder_pid() -> int | None:
    ws = NSWorkspace.sharedWorkspace()
    for app in ws.runningApplications():
        bid = app.bundleIdentifier()
        if bid == "com.apple.finder":
            return app.processIdentifier()
    return None


_MIN_WINDOW_AREA = 50 * 50


def _get_onscreen_windows() -> list[dict]:
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
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
            "pid": int(
                w.get("kCGWindowOwnerPID", 0),
            ),
            "x": x, "y": y,
            "w": width, "h": height,
            "owner": str(
                w.get("kCGWindowOwnerName", "?"),
            ),
        })
    return result


def _get_visible_app_pids(
        exclude_pids: set[int],
        windows: list[dict],
) -> list[tuple[int, str]]:
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    for w in windows:
        pid = w["pid"]
        if pid in exclude_pids or pid in seen:
            continue
        seen.add(pid)
        result.append((pid, w["owner"]))
    return result


def _is_occluded(
        cx: int, cy: int,
        elem_pid: int,
        windows: list[dict],
) -> bool:
    for w in windows:
        if w["pid"] == elem_pid:
            return False
        if (
                w["x"] <= cx <= w["x"] + w["w"]
                and w["y"] <= cy <= w["y"] + w["h"]
        ):
            return True
    return False


_RAW_LIMIT = 5000


def get_accessibility_elements() -> (
        tuple[list[UIElement], float]
):
    """Collect UI elements from the accessibility tree."""
    app_raw: list[dict] = []

    pid, app_name = _get_frontmost_pid()
    if pid is not None:
        app_ref = AXUIElementCreateApplication(pid)
        _ensure_accessibility(app_ref)

        ctx_menu = _find_context_menu(app_ref)
        if ctx_menu is not None:
            _walk_tree(
                ctx_menu, app_raw, _RAW_LIMIT, depth=1,
            )

        focused_win = _ax_attr(
            app_ref, "AXFocusedWindow",
        )
        if focused_win:
            children = (
                    _ax_attr(focused_win, "AXChildren")
                    or []
            )
            for child in children:
                if len(app_raw) >= _RAW_LIMIT:
                    break
                child_role = (
                        _ax_attr(child, "AXRole") or ""
                )
                if child_role in _DIALOG_ROLES:
                    _walk_tree(
                        child, app_raw, _RAW_LIMIT,
                        depth=1,
                    )
            _walk_tree(
                focused_win, app_raw, _RAW_LIMIT,
                depth=1,
            )

        windows = _ax_attr(app_ref, "AXWindows")
        if windows:
            for win in windows:
                if len(app_raw) >= _RAW_LIMIT:
                    break
                _walk_tree(
                    win, app_raw, _RAW_LIMIT, depth=1,
                )

        menu_bar = _ax_attr(app_ref, "AXMenuBar")
        if menu_bar and len(app_raw) < _RAW_LIMIT:
            _walk_tree(
                menu_bar, app_raw, _RAW_LIMIT, depth=1,
            )

        if len(app_raw) < _RAW_LIMIT:
            _walk_tree(app_ref, app_raw, _RAW_LIMIT)

    onscreen_windows = _get_onscreen_windows()

    dock_pid = _get_dock_pid()
    finder_pid = _get_finder_pid()
    exclude_pids: set[int] = set()
    if pid is not None:
        exclude_pids.add(pid)
    if dock_pid is not None:
        exclude_pids.add(dock_pid)
    if finder_pid is not None:
        exclude_pids.add(finder_pid)

    bg_pids = _get_visible_app_pids(
        exclude_pids, onscreen_windows,
    )

    for bg_pid, bg_name in bg_pids:
        bg_ref = AXUIElementCreateApplication(bg_pid)
        windows = _ax_attr(bg_ref, "AXWindows")
        if not windows:
            continue
        for win in windows:
            pos = _ax_position(win)
            size = _ax_size(win)
            if (
                    not pos
                    or not size
                    or size[0] < _MIN_SIZE
                    or size[1] < _MIN_SIZE
            ):
                continue
            label = _build_label(win)
            app_raw.append({
                "role": "AXWindow",
                "subrole": str(
                    _ax_attr(win, "AXSubrole") or "",
                ),
                "label": label,
                "x": pos[0],
                "y": pos[1],
                "w": size[0],
                "h": size[1],
                "states": _collect_states(
                    win, "AXWindow",
                ),
                "path": "AXWindow",
                "depth": 0,
                "source": bg_name,
                "pid": bg_pid,
                "labeled": False,
            })

    finder_raw: list[dict] = []
    if finder_pid is not None and finder_pid != pid:
        finder_ref = AXUIElementCreateApplication(
            finder_pid,
        )
        _walk_tree(
            finder_ref, finder_raw, _RAW_LIMIT,
        )
        for item in finder_raw:
            item["source"] = "Finder"
            item["pid"] = finder_pid

    dock_raw: list[dict] = []
    if dock_pid is not None:
        dock_ref = AXUIElementCreateApplication(
            dock_pid,
        )
        _walk_tree(dock_ref, dock_raw, _RAW_LIMIT)

    raw = app_raw + finder_raw + dock_raw

    backing_scale = _get_backing_scale()
    sw, sh = _get_screen_point_size()

    seen: set[tuple[int, int, int, int]] = set()
    visible: list[UIElement] = []

    for item in raw:
        x = item["x"]
        y = item["y"]
        w = item["w"]
        h = item["h"]

        if not _overlaps_screen(x, y, w, h, sw, sh):
            continue

        is_occluded_item = False
        if item.get("source") and onscreen_windows:
            cx, cy = x + w // 2, y + h // 2
            if _is_occluded(
                    cx, cy, item["pid"], onscreen_windows,
            ):
                is_occluded_item = True

        key = (x, y, w, h)
        if key in seen:
            continue
        seen.add(key)

        states = list(item.get("states", []))
        labeled = item.get("labeled", True)
        if is_occluded_item:
            states.append("OCCLUDED")
            labeled = False

        if (
                not labeled
                and item["role"] not in _UNLABELED_ROLES
                and "OFFSCREEN" not in states
                and "OCCLUDED" not in states
        ):
            states.append("HIDDEN")

        visible.append(UIElement(
            id=len(visible),
            role=item["role"],
            label=item["label"],
            x=x, y=y, w=w, h=h,
            states=states,
            source=item.get("source", ""),
            subrole=item.get("subrole", ""),
            path=item.get("path", ""),
            depth=item.get("depth", 0),
            labeled=labeled,
        ))

    return visible, backing_scale
