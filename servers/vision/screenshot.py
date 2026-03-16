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


def capture_screenshot(
        target_width: int = DEFAULT_WIDTH,
) -> tuple[str, int, int, float]:
    """Returns (b64_jpeg, width, height, scale)."""
    screenshot = _grab_screen()
    real_w, real_h = screenshot.size

    scale = real_w / target_width
    new_h = round(real_h / scale)

    scaled = screenshot.resize(
        (target_width, new_h), Image.LANCZOS,
    )

    buf = io.BytesIO()
    scaled.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return b64, target_width, new_h, scale
