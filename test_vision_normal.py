"""Quick test: screenshot -> accessibility -> save labeled image + elements."""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from servers.vision.accessibility import get_accessibility_elements
    from servers.vision.labeler import draw_labels
    from servers.vision.overlay import get_overlay_elements
    from servers.vision.screenshot import capture_screenshot

    print("Waiting 5 seconds ...")
    time.sleep(5)

    print("Capturing screenshot ...")
    t0 = time.time()
    b64, img_w, img_h, scale, backing_scale = capture_screenshot()
    print(f"  {img_w}x{img_h}  scale={scale:.2f}  backing={backing_scale:.1f}  ({time.time() - t0:.2f}s)")

    print("Collecting accessibility elements ...")
    t0 = time.time()
    elements, _ = get_accessibility_elements()
    overlay_els = get_overlay_elements(elements)
    if overlay_els:
        elements = list(elements) + overlay_els
        for i, el in enumerate(elements):
            el.id = i
    print(f"  {len(elements)} elements ({time.time() - t0:.2f}s)")

    print("Drawing labels ...")
    t0 = time.time()
    labeled_b64 = draw_labels(
        b64,
        [el.to_dict() for el in elements if el.labeled],
        scale,
        backing_scale,
    )
    print(f"  done ({time.time() - t0:.2f}s)")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("debug_logs", f"test_vision_normal_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "screenshot_raw.png"), "wb") as f:
        f.write(base64.b64decode(b64))

    with open(os.path.join(out_dir, "screenshot_labeled.png"), "wb") as f:
        f.write(base64.b64decode(labeled_b64))

    element_lines = [el.format_entry() for el in elements]
    with open(os.path.join(out_dir, "elements.txt"), "w") as f:
        f.write("\n".join(element_lines))

    positions = [el.to_position() for el in elements]
    with open(os.path.join(out_dir, "element_positions.json"), "w") as f:
        json.dump(positions, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  screenshot_raw.png        (what was captured)")
    print(f"  screenshot_labeled.png    (with bounding boxes)")
    print(f"  elements.txt              ({len(element_lines)} elements)")
    print(f"  element_positions.json    (id/x/y/label/role)")


if __name__ == "__main__":
    main()
