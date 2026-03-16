"""Overlay detection for AX-invisible transient UI."""

from __future__ import annotations

import logging

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

from servers.vision.accessibility import UIElement
from servers.vision.ocr import (
    capture_window_image,
    ocr_image,
)

logger = logging.getLogger(__name__)

_OVERLAY_LAYERS = {3, 8, 101}
_MAX_SYSTEM_LAYER = 1_000_000
_MIN_OVERLAY_AREA = 40 * 20

_ROLE_FOR_LAYER: dict[int, str] = {
    101: "AXMenuItem",
    3: "AXMenuItem",
}
_DEFAULT_OVERLAY_ROLE = "AXStaticText"

_AX_OVERLAY_ROLES = {
    "AXMenuItem",
    "AXMenu",
    "AXMenuButton",
    "AXSheet",
    "AXDialog",
}


def _detect_overlay_windows() -> list[dict]:
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
    if not windows:
        return []

    overlays: list[dict] = []
    for w in windows:
        layer = int(w.get("kCGWindowLayer", 0))
        if (
                layer not in _OVERLAY_LAYERS
                or layer >= _MAX_SYSTEM_LAYER
        ):
            continue

        bounds = w.get("kCGWindowBounds", {})
        x = int(bounds.get("X", 0))
        y = int(bounds.get("Y", 0))
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))

        if width * height < _MIN_OVERLAY_AREA:
            continue

        overlays.append({
            "pid": int(
                w.get("kCGWindowOwnerPID", 0),
            ),
            "window_id": int(
                w.get("kCGWindowNumber", 0),
            ),
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "layer": layer,
            "owner": str(
                w.get("kCGWindowOwnerName", "?"),
            ),
        })

    return overlays


def _is_covered(
        overlay: dict, elements: list[UIElement],
) -> bool:
    ox = overlay["x"]
    oy = overlay["y"]
    ow = overlay["w"]
    oh = overlay["h"]
    for el in elements:
        if el.role not in _AX_OVERLAY_ROLES:
            continue
        cx, cy = el.center
        if (
                ox <= cx <= ox + ow
                and oy <= cy <= oy + oh
        ):
            return True
    return False


def get_overlay_elements(
        existing_elements: list[UIElement],
) -> list[UIElement]:
    """Detect overlay windows and return OCR elements."""
    overlays = _detect_overlay_windows()
    if not overlays:
        return []

    new_elements: list[UIElement] = []
    next_id = len(existing_elements)

    for ov in overlays:
        if _is_covered(ov, existing_elements):
            continue

        cg_image = capture_window_image(
            ov["window_id"],
        )
        if cg_image is None:
            continue

        role = _ROLE_FOR_LAYER.get(
            ov["layer"], _DEFAULT_OVERLAY_ROLE,
        )
        items = ocr_image(cg_image, ov, role)

        for item in items:
            new_elements.append(UIElement(
                id=next_id + len(new_elements),
                role=item["role"],
                label=item["label"],
                x=item["x"],
                y=item["y"],
                w=item["w"],
                h=item["h"],
                states=item["states"],
            ))

    return new_elements
