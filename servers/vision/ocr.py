"""macOS native OCR via the Vision framework (VNRecognizeTextRequest).

Provides reusable text recognition for any CGImage, returning element
dicts with point-space coordinates ready for UIElement construction.
"""

from __future__ import annotations

import logging
from typing import Any

from Quartz import (
    CGWindowListCreateImage,
    CGRectNull,
    kCGWindowListOptionIncludingWindow,
    kCGWindowImageBoundsIgnoreFraming,
)

logger = logging.getLogger(__name__)


def capture_window_image(window_id: int) -> Any | None:
    """Capture a CGImage of a single window by its CGWindowList ID."""
    cg_image = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )
    return cg_image


def ocr_image(cg_image: Any, bounds: dict, role: str = "AXStaticText") -> list[dict]:
    """Run macOS Vision OCR on a CGImage, return element dicts in point-space.

    Parameters
    ----------
    cg_image : A Quartz CGImageRef.
    bounds : dict with keys x, y, w, h (point-space window bounds).
    role : AX role to assign to each detected text element.

    Returns
    -------
    list of dicts: {role, label, x, y, w, h, states}, sorted top-to-bottom.
    """
    try:
        import Vision
    except ImportError:
        logger.warning("pyobjc-framework-Vision not installed; OCR disabled")
        return []

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None,
    )
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    success, error = handler.performRequests_error_([request], None)
    if not success:
        logger.warning("Vision OCR failed: %s", error)
        return []

    observations = request.results()
    if not observations:
        return []

    wx, wy = bounds["x"], bounds["y"]
    ww, wh = bounds["w"], bounds["h"]

    items: list[dict] = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        if not text:
            continue

        bbox = obs.boundingBox()
        # Vision coords: normalised 0-1, origin at bottom-left.
        # Convert to point-space with top-left origin.
        py = wy + (1.0 - bbox.origin.y - bbox.size.height) * wh
        ph = bbox.size.height * wh

        if ph < 5:
            continue

        items.append({
            "role": role,
            "label": text,
            "x": wx,
            "y": int(py),
            "w": ww,
            "h": max(int(ph), 10),
            "states": [],
        })

    items.sort(key=lambda d: d["y"])

    logger.info("OCR found %d text items in region (%d,%d %dx%d)", len(items), wx, wy, ww, wh)
    return items
