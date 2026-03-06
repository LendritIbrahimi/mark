"""Draw numbered bounding boxes on a screenshot image.

Uses bright, high-contrast colors with role-matched tag backgrounds
and collision avoidance, optimized for vision model readability.
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

BOX_WIDTH = 3
FILL_ALPHA = 25
FONT_SIZE = 14
TAG_PAD = 3


def _get_color(role: str) -> str:
    return ROLE_COLORS.get(role, DEFAULT_COLOR)


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _tag_text_color(hex_color: str) -> str:
    """Black or white text depending on tag background luminance."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else "#FFFFFF"


def _load_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _find_tag_position(
    tw: int,
    th: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
    placed: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Find a non-overlapping position for a tag near its bounding box."""
    tag_w = tw + TAG_PAD * 2
    tag_h = th + TAG_PAD * 2

    candidates = [
        (max(0, bx1), max(0, by1 - tag_h)),
        (max(0, bx1), min(img_h - tag_h, by2)),
        (min(img_w - tag_w, bx2), max(0, by1 - tag_h)),
        (bx1 + BOX_WIDTH, by1 + BOX_WIDTH),
    ]

    for lx, ly in candidates:
        rect = (lx, ly, lx + tag_w, ly + tag_h)
        if not any(_rects_overlap(rect, p) for p in placed):
            return rect

    lx, ly = candidates[0]
    return (lx, ly, lx + tag_w, ly + tag_h)


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
    font = _load_font()

    img_w, img_h = img.size
    placed_tags: list[tuple[int, int, int, int]] = []

    for el in elements:
        color = _get_color(el["role"])

        ix = int(el["x"] * backing_scale / scale)
        iy = int(el["y"] * backing_scale / scale)
        iw = int(el["w"] * backing_scale / scale)
        ih = int(el["h"] * backing_scale / scale)

        if ix + iw < 0 or iy + ih < 0 or ix >= img_w or iy >= img_h:
            continue

        bx1 = max(0, ix)
        by1 = max(0, iy)
        bx2 = min(img_w - 1, ix + iw)
        by2 = min(img_h - 1, iy + ih)

        fill = _hex_to_rgba(color, FILL_ALPHA)
        draw.rectangle([bx1, by1, bx2, by2], fill=fill, outline=color, width=BOX_WIDTH)

        tag = str(el["id"])
        bbox = font.getbbox(tag)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        tag_rect = _find_tag_position(
            tw, th, bx1, by1, bx2, by2, placed_tags, img_w, img_h,
        )
        placed_tags.append(tag_rect)

        draw.rectangle(list(tag_rect), fill=color)
        text_color = _tag_text_color(color)
        draw.text(
            (tag_rect[0] + TAG_PAD, tag_rect[1] + TAG_PAD),
            tag, fill=text_color, font=font,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    labeled_b64 = base64.b64encode(buf.getvalue()).decode()

    logger.debug("Labeled %d elements", len(elements))
    return labeled_b64
