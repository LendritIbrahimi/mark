"""Vision MCP Server -- screenshot with labeled boxes."""

from __future__ import annotations

import json
import logging

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyProhibited,
)

NSApplication.sharedApplication().setActivationPolicy_(
    NSApplicationActivationPolicyProhibited,
)

logging.basicConfig(level=logging.WARNING)

from fastmcp import FastMCP

from servers.vision.accessibility import (
    get_accessibility_elements,
)
from servers.vision.labeler import draw_labels
from servers.vision.overlay import get_overlay_elements
from servers.vision.screenshot import capture_screenshot

mcp = FastMCP("mark-vision")


@mcp.tool()
def observe() -> str:
    """Capture screenshot with labeled bounding boxes."""
    b64, img_w, img_h, scale = capture_screenshot()

    elements, backing_scale = (
        get_accessibility_elements()
    )

    overlay_els = get_overlay_elements(elements)
    if overlay_els:
        elements = list(elements) + overlay_els
        for i, el in enumerate(elements):
            el.id = i

    labeled_b64 = draw_labels(
        b64,
        [el.to_dict() for el in elements if el.labeled],
        scale,
        backing_scale,
    )

    element_lines = [
        el.format_entry() for el in elements
    ]

    return json.dumps({
        "image": labeled_b64,
        "elements": "\n".join(element_lines),
        "element_positions": [
            el.to_position() for el in elements
        ],
        "scale": scale,
        "backing_scale": backing_scale,
    })


if __name__ == "__main__":
    mcp.run()
