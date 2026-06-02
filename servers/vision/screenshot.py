"""Screenshot capture and scaling for macOS."""

from __future__ import annotations

import base64
import io

import AppKit
from PIL import Image
from Quartz import (
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectInfinite,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowImageDefault,
    kCGWindowListOptionOnScreenOnly,
)

DEFAULT_WIDTH = 1280


def _grab_screen() -> Image.Image:
    """Capture the screen via CoreGraphics (in-process, honours TCC grant)."""
    cg_img = CGWindowListCreateImage(
        CGRectInfinite,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        kCGWindowImageDefault,
    )
    if cg_img is None:
        raise RuntimeError(
            "CGWindowListCreateImage returned None — "
            "grant Screen Recording permission to mark.app in "
            "System Settings → Privacy & Security → Screen Recording"
        )
    ns_img = AppKit.NSImage.alloc().initWithCGImage_size_(cg_img, AppKit.NSZeroSize)
    tiff = ns_img.TIFFRepresentation()
    rep = AppKit.NSBitmapImageRep.imageRepWithData_(tiff)
    png = rep.representationUsingType_properties_(
        AppKit.NSBitmapImageFileTypePNG, None,
    )
    return Image.open(io.BytesIO(bytes(png))).convert("RGB")


def get_backing_scale() -> float:
    """Current display backing scale (2.0 on Retina, 1.0 otherwise)."""
    screen = AppKit.NSScreen.mainScreen()
    if screen is None:
        return 2.0
    return float(screen.backingScaleFactor())


def capture_screenshot(
        target_width: int = DEFAULT_WIDTH,
        image_format: str = "jpeg",
) -> tuple[str, int, int, float, float]:
    """Returns (b64_image, width, height, scale, backing_scale).

    Args:
        target_width: Resize the screenshot to this width (aspect ratio preserved).
        image_format: ``"jpeg"`` (smaller, lossy) or ``"png"`` (lossless, better
            for OCR / OmniParser).
    """
    backing_scale = get_backing_scale()
    screenshot = _grab_screen()
    real_w, real_h = screenshot.size

    scale = real_w / target_width
    new_h = round(real_h / scale)

    scaled = screenshot.resize(
        (target_width, new_h), Image.LANCZOS,
    )

    buf = io.BytesIO()
    if image_format == "png":
        scaled.save(buf, format="PNG")
    else:
        scaled.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return b64, target_width, new_h, scale, backing_scale
