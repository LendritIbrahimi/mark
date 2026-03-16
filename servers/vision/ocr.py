"""macOS native OCR via the Vision framework."""

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
    return CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )


def ocr_image(
        cg_image: Any,
        bounds: dict,
        role: str = "AXStaticText",
) -> list[dict]:
    """Run Vision OCR on a CG image and return elements."""
    try:
        import Vision
    except ImportError:
        logger.warning(
            "pyobjc-framework-Vision not installed; "
            "OCR disabled",
        )
        return []

    handler = (
        Vision.VNImageRequestHandler
        .alloc()
        .initWithCGImage_options_(cg_image, None)
    )
    request = (
        Vision.VNRecognizeTextRequest.alloc().init()
    )
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelAccurate,
    )
    request.setUsesLanguageCorrection_(True)

    success, error = handler.performRequests_error_(
        [request], None,
    )
    if not success:
        logger.error("Vision OCR failed: %s", error)
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
        py = (
                wy
                + (1.0 - bbox.origin.y - bbox.size.height)
                * wh
        )
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
    return items
