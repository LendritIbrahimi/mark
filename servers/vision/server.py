"""Vision MCP Server -- screenshot with labeled boxes."""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

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
    b64, img_w, img_h, scale, backing_scale = capture_screenshot()

    elements, _ = (
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


@mcp.tool()
def observe_omniparser() -> str:
    """Capture screenshot and detect UI elements via local OmniParser server."""
    omni_url = os.environ.get("OMNIPARSER_LOCAL_URL", "").rstrip("/")
    if not omni_url:
        raise RuntimeError(
            "OMNIPARSER_LOCAL_URL is not set. "
            "Start the OmniParser server before running mark with --omniparser."
        )

    max_retries = 3
    for attempt in range(max_retries):
        b64, img_w, img_h, scale, backing_scale = capture_screenshot(
            target_width=1920, image_format="jpeg",
        )
        try:
            resp = httpx.post(
                f"{omni_url}/parse/",
                json={"base64_image": b64},
                timeout=600,
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" in result or not result.get("parsed_content_list"):
                text = result.get("error", result.get("detail", ""))
                if "timed out" in str(text).lower() and attempt < max_retries - 1:
                    logging.getLogger(__name__).warning(
                        "OmniParser timed out (attempt %d/%d), retrying...",
                        attempt + 1, max_retries,
                    )
                    time.sleep(2)
                    continue
            break
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to OmniParser server at {omni_url}. "
                "Make sure the server is running."
            )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            if "timed out" in body.lower() and attempt < max_retries - 1:
                logging.getLogger(__name__).warning(
                    "OmniParser upstream timeout (attempt %d/%d), retrying...",
                    attempt + 1, max_retries,
                )
                time.sleep(2)
                continue
            raise

    annotated_image = result.get("som_image_base64", b64)
    parsed = result.get("parsed_content_list", [])

    element_positions = []
    element_lines = []
    for i, el in enumerate(parsed):
        bbox = el.get("bbox", [])
        try:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cx = int(round(((x1 + x2) / 2 * img_w) * scale / backing_scale))
                cy = int(round(((y1 + y2) / 2 * img_h) * scale / backing_scale))
            else:
                cx, cy = 0, 0
        except (TypeError, ValueError):
            cx, cy = 0, 0

        content = el.get("content", "") or ""
        el_type = el.get("type", "unknown")
        label = "Icon" if "icon" in el_type.lower() else "Text"

        element_positions.append({
            "id": i,
            "x": cx,
            "y": cy,
            "label": content,
            "role": el_type,
        })
        element_lines.append(f"[{i}] {label}: \"{content}\"")

    return json.dumps({
        "image": annotated_image,
        "elements": "\n".join(element_lines),
        "element_positions": element_positions,
        "scale": scale,
        "backing_scale": backing_scale,
    })


if __name__ == "__main__":
    mcp.run()
