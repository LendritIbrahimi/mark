"""Draw numbered bounding boxes on a screenshot image.

Uses bright, high-contrast colors optimized for vision model readability.
Each element gets a colored rectangle and a clearly visible ID tag.
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ROLE_COLORS: dict[str, str] = {
    "AXButton": "#FF2D2D",
    "AXTextField": "#00E639",
    "AXTextArea": "#00E639",
    "AXLink": "#FF8C00",
    "AXCheckBox": "#E040FB",
    "AXRadioButton": "#E040FB",
    "AXMenuItem": "#00BCD4",
    "AXMenuBarItem": "#00BCD4",
    "AXPopUpButton": "#FFEA00",
    "AXComboBox": "#FFEA00",
    "AXTab": "#76FF03",
    "AXStaticText": "#40C4FF",
    "AXImage": "#FF6FD8",
    "AXHeading": "#FF9100",
    "AXSlider": "#B388FF",
    "AXDockItem": "#00E5FF",
    "AXSheet": "#FFD600",
    "AXDialog": "#FFD600",
}
DEFAULT_COLOR = "#FF2D2D"

TAG_BG = "#FF0000"
TAG_FG = "#FFFFFF"
BOX_WIDTH = 2


def _get_color(role: str) -> str:
    return ROLE_COLORS.get(role, DEFAULT_COLOR)


def _load_font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def draw_labels(
    image_b64: str,
    elements: list[dict],
    scale: float,
    backing_scale: float = 1.0,
) -> str:
    """Draw bounding boxes and ID tags on the screenshot.

    Parameters
    ----------
    image_b64 : base64-encoded JPEG of the scaled screenshot.
    elements : list of dicts with keys id, role, x, y, w, h
               (point-space coordinates).
    scale : screenshot_pixel_width / target_width.
    backing_scale : Retina factor (2.0 on HiDPI, 1.0 otherwise).
               Used to convert point-space coords to image-space:
               image_coord = point_coord * backing_scale / scale.
    """
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(13)

    img_w, img_h = img.size

    for el in elements:
        color = _get_color(el["role"])

        ix = int(el["x"] * backing_scale / scale)
        iy = int(el["y"] * backing_scale / scale)
        iw = int(el["w"] * backing_scale / scale)
        ih = int(el["h"] * backing_scale / scale)

        if ix + iw < 0 or iy + ih < 0 or ix >= img_w or iy >= img_h:
            continue

        cx = max(0, ix)
        cy = max(0, iy)
        cx2 = min(img_w - 1, ix + iw)
        cy2 = min(img_h - 1, iy + ih)
        draw.rectangle([cx, cy, cx2, cy2], outline=color, width=BOX_WIDTH)

        tag = str(el["id"])
        bbox = font.getbbox(tag)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3

        lx = max(0, cx)
        ly = max(0, cy - th - pad * 2)
        draw.rectangle(
            [lx, ly, lx + tw + pad * 2, ly + th + pad * 2],
            fill=TAG_BG,
        )
        draw.text((lx + pad, ly + pad), tag, fill=TAG_FG, font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    labeled_b64 = base64.b64encode(buf.getvalue()).decode()

    logger.debug("Labeled %d elements", len(elements))
    return labeled_b64
