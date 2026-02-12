"""Vision MCP Server -- screenshot with labeled bounding boxes."""

from __future__ import annotations

import json
import logging

from AppKit import NSApplication, NSApplicationActivationPolicyProhibited
NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyProhibited)

from fastmcp import FastMCP

from servers.vision.screenshot import capture_screenshot
from servers.vision.accessibility import get_accessibility_elements
from servers.vision.labeler import draw_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

mcp = FastMCP("mark-vision")

SCREENSHOT_WIDTH = 1280
MAX_ELEMENTS = 150


@mcp.tool()
def observe(
    width: int = SCREENSHOT_WIDTH,
    max_elements: int = MAX_ELEMENTS,
) -> str:
    """Capture screenshot with labeled bounding boxes and an element list.

    Returns JSON:
      image             -- base64 JPEG with numbered bounding boxes
      elements          -- human-readable element list, e.g. [3] Button: "Submit"
      element_positions -- [{id, x, y}] center coords for action resolution
      scale             -- uniform factor mapping image coords to pixel coords
    """
    b64, img_w, img_h, scale = capture_screenshot(target_width=width)
    elements, backing_scale = get_accessibility_elements(max_elements=max_elements)

    labeled_b64 = draw_labels(
        b64, [el.to_dict() for el in elements], scale, backing_scale,
    )

    return json.dumps({
        "image": labeled_b64,
        "elements": "\n".join(el.format_entry() for el in elements),
        "element_positions": [el.to_position() for el in elements],
        "scale": scale,
        "backing_scale": backing_scale,
    })


if __name__ == "__main__":
    mcp.run()
