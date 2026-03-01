"""OmniParser v2 element detection via HuggingFace Inference Endpoint.

Alternative to the accessibility tree for detecting UI elements.  Sends the
screenshot to a remote OmniParser v2 instance and converts the response into
the same UIElement format used by the rest of the pipeline.

Required environment variables:
    OMNIPARSER_URL   -- Base URL of the HuggingFace endpoint
    OMNIPARSER_TOKEN -- HuggingFace API token

The endpoint must accept a Gradio-style ``/api/predict`` call and return
parsed elements with bounding boxes.  See the README for the expected
request/response format.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any

from servers.vision.accessibility import UIElement

logger = logging.getLogger(__name__)

_URL = os.environ.get("OMNIPARSER_URL", "")
_TOKEN = os.environ.get("OMNIPARSER_TOKEN", "")

_ELEMENT_TYPE_TO_ROLE: dict[str, str] = {
    "button": "AXButton",
    "text": "AXStaticText",
    "link": "AXLink",
    "input": "AXTextField",
    "image": "AXImage",
    "icon": "AXButton",
    "checkbox": "AXCheckBox",
    "dropdown": "AXPopUpButton",
    "tab": "AXTab",
    "slider": "AXSlider",
    "heading": "AXHeading",
}
_DEFAULT_ROLE = "AXStaticText"


def _post_to_endpoint(screenshot_b64: str) -> dict:
    """Send the screenshot to the OmniParser HF endpoint and return raw JSON."""
    if not _URL:
        raise RuntimeError("OMNIPARSER_URL not set in environment")

    url = _URL.rstrip("/") + "/api/predict"
    data_uri = f"data:image/jpeg;base64,{screenshot_b64}"

    payload = json.dumps({
        "data": [data_uri, 0.05, 0.1, True, 640],
    }).encode()

    headers = {
        "Content-Type": "application/json",
    }
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        logger.error("OmniParser request failed: %s", exc)
        raise


def _parse_icon_list(text: str) -> list[dict]:
    """Parse the 'icon N: description' text format from OmniParser.

    Returns list of {id, label} dicts (no bbox -- caller must merge with
    coordinate data separately).
    """
    items: list[dict] = []
    for line in text.strip().splitlines():
        m = re.match(r"icon\s+(\d+):\s*(.*)", line, re.IGNORECASE)
        if m:
            items.append({"id": int(m.group(1)), "label": m.group(2).strip()})
    return items


def _parse_response(
    response: dict,
    img_w: int,
    img_h: int,
    scale: float,
    backing_scale: float,
) -> list[UIElement]:
    """Convert OmniParser response into UIElement list.

    Supports two response formats:

    1. Structured JSON:  {"elements": [{"label", "bbox", "type"}, ...]}
       bbox is [x1, y1, x2, y2] normalised 0-1.

    2. Gradio-style:    {"data": [labeled_image, parsed_text]}
       plus optional    {"data": [labeled_image, parsed_text, coordinates_json]}
       where coordinates_json maps icon id -> [x1, y1, x2, y2] in 0-1.
    """
    factor = scale / backing_scale

    # Format 1: structured JSON with bboxes
    if "elements" in response:
        elements: list[UIElement] = []
        for i, el in enumerate(response["elements"]):
            bbox = el.get("bbox", [0, 0, 0, 0])
            role = _ELEMENT_TYPE_TO_ROLE.get(el.get("type", ""), _DEFAULT_ROLE)
            label = el.get("label", "")

            px = int(bbox[0] * img_w * factor)
            py = int(bbox[1] * img_h * factor)
            pw = int((bbox[2] - bbox[0]) * img_w * factor)
            ph = int((bbox[3] - bbox[1]) * img_h * factor)

            if pw < 5 or ph < 5:
                continue

            elements.append(UIElement(
                id=i, role=role, label=label,
                x=px, y=py, w=pw, h=ph, states=[],
            ))
        return elements

    # Format 2: Gradio-style text + optional coordinates
    data = response.get("data", [])
    if len(data) < 2:
        logger.warning("OmniParser response has no usable data")
        return []

    parsed_text = data[1] if isinstance(data[1], str) else ""
    icon_items = _parse_icon_list(parsed_text)

    coords: dict[int, list[float]] = {}
    if len(data) >= 3 and isinstance(data[2], (dict, str)):
        raw = data[2] if isinstance(data[2], dict) else json.loads(data[2])
        coords = {int(k): v for k, v in raw.items()}

    elements = []
    for item in icon_items:
        iid = item["id"]
        label = item["label"]

        if iid not in coords:
            continue

        bbox = coords[iid]
        px = int(bbox[0] * img_w * factor)
        py = int(bbox[1] * img_h * factor)
        pw = int((bbox[2] - bbox[0]) * img_w * factor)
        ph = int((bbox[3] - bbox[1]) * img_h * factor)

        if pw < 5 or ph < 5:
            continue

        elements.append(UIElement(
            id=len(elements), role=_DEFAULT_ROLE, label=label,
            x=px, y=py, w=pw, h=ph, states=[],
        ))

    return elements


def get_omniparser_elements(
    screenshot_b64: str,
    img_w: int,
    img_h: int,
    scale: float,
    backing_scale: float,
) -> list[UIElement]:
    """Send screenshot to OmniParser and return detected UI elements.

    Parameters
    ----------
    screenshot_b64 : Base64-encoded JPEG of the scaled screenshot.
    img_w, img_h : Dimensions of the screenshot image in pixels.
    scale : Factor mapping image pixels to real screen pixels.
    backing_scale : Retina backing scale factor.

    Returns
    -------
    list[UIElement] with point-space coordinates matching pyautogui.
    """
    logger.info("Calling OmniParser endpoint...")
    response = _post_to_endpoint(screenshot_b64)
    elements = _parse_response(response, img_w, img_h, scale, backing_scale)
    logger.info("OmniParser returned %d elements", len(elements))
    return elements
