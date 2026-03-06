"""Screenshot capture and scaling for macOS."""

from __future__ import annotations

import base64
import io
import logging

import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)


def capture_screenshot(target_width: int = 1280) -> tuple[str, int, int, float]:
    """Capture the screen, scale to *target_width*, return as base64 JPEG.

    Returns
    -------
    (base64_jpeg, scaled_width, scaled_height, scale)
        scale maps from image coords to real pixel coords (uniform, no warping).
    """
    screenshot = pyautogui.screenshot().convert("RGB")
    real_w, real_h = screenshot.size

    scale = real_w / target_width
    new_h = round(real_h / scale)

    scaled = screenshot.resize((target_width, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    scaled.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()

    logger.debug(
        "Screenshot: %dx%d -> %dx%d (scale %.2f)",
        real_w, real_h, target_width, new_h, scale,
    )
    return b64, target_width, new_h, scale
