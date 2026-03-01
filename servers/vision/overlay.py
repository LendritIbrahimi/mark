"""Overlay detection for AX-invisible transient UI (context menus, popovers, etc).

Many macOS apps render context menus and popovers as separate windows that
don't appear in their accessibility tree.  This module uses CGWindowList to
detect those elevated-layer windows, then delegates to the OCR module to
extract clickable elements.
"""

from __future__ import annotations

import logging

from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)

from servers.vision.accessibility import UIElement
from servers.vision.ocr import capture_window_image, ocr_image

logger = logging.getLogger(__name__)

_OVERLAY_LAYERS = {3, 8, 101}
_MAX_SYSTEM_LAYER = 1_000_000
_MIN_OVERLAY_AREA = 40 * 20

_ROLE_FOR_LAYER: dict[int, str] = {
    101: "AXMenuItem",
    3: "AXMenuItem",
}
_DEFAULT_OVERLAY_ROLE = "AXStaticText"

_AX_OVERLAY_ROLES = {"AXMenuItem", "AXMenu", "AXMenuButton", "AXSheet", "AXDialog"}


# ---------------------------------------------------------------------------
# CGWindowList overlay detection
# ---------------------------------------------------------------------------


def _detect_overlay_windows() -> list[dict]:
    """Find on-screen windows at overlay layers via CGWindowList.

    Returns list of dicts: {pid, window_id, x, y, w, h, layer, owner}.
    Coordinates are in point-space.
    """
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID,
    )
    if not windows:
        return []

    overlays: list[dict] = []
    for w in windows:
        layer = int(w.get("kCGWindowLayer", 0))
        if layer not in _OVERLAY_LAYERS or layer >= _MAX_SYSTEM_LAYER:
            continue

        bounds = w.get("kCGWindowBounds", {})
        x = int(bounds.get("X", 0))
        y = int(bounds.get("Y", 0))
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))

        if width * height < _MIN_OVERLAY_AREA:
            continue

        overlays.append({
            "pid": int(w.get("kCGWindowOwnerPID", 0)),
            "window_id": int(w.get("kCGWindowNumber", 0)),
            "x": x, "y": y, "w": width, "h": height,
            "layer": layer,
            "owner": str(w.get("kCGWindowOwnerName", "?")),
        })

    return overlays


# ---------------------------------------------------------------------------
# Coverage check -- skip OCR when AX already found the overlay
# ---------------------------------------------------------------------------


def _is_covered(overlay: dict, elements: list[UIElement]) -> bool:
    """True if AX elements already describe this overlay's content."""
    ox, oy, ow, oh = overlay["x"], overlay["y"], overlay["w"], overlay["h"]
    for el in elements:
        if el.role not in _AX_OVERLAY_ROLES:
            continue
        cx, cy = el.center
        if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_overlay_elements(
    existing_elements: list[UIElement],
) -> list[UIElement]:
    """Detect overlay windows missing from the AX tree and OCR their content.

    Returns UIElements whose IDs continue after *existing_elements*.
    Caller should re-number all IDs after merging if needed.
    """
    overlays = _detect_overlay_windows()
    if not overlays:
        return []

    logger.debug("Detected %d overlay window(s)", len(overlays))

    new_elements: list[UIElement] = []
    next_id = len(existing_elements)

    for ov in overlays:
        if _is_covered(ov, existing_elements):
            logger.debug(
                "Overlay at (%d,%d) %dx%d already covered by AX",
                ov["x"], ov["y"], ov["w"], ov["h"],
            )
            continue

        cg_image = capture_window_image(ov["window_id"])
        if cg_image is None:
            logger.debug("Failed to capture window %d", ov["window_id"])
            continue

        role = _ROLE_FOR_LAYER.get(ov["layer"], _DEFAULT_OVERLAY_ROLE)
        items = ocr_image(cg_image, ov, role)

        for item in items:
            new_elements.append(UIElement(
                id=next_id + len(new_elements),
                role=item["role"],
                label=item["label"],
                x=item["x"], y=item["y"],
                w=item["w"], h=item["h"],
                states=item["states"],
            ))

    if new_elements:
        logger.info("Added %d overlay element(s) via OCR", len(new_elements))

    return new_elements
