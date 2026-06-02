"""Quick test: screenshot -> OmniParser -> save annotated image + elements."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    omni_url = f"http://127.0.0.1:{port}"
    os.environ["OMNIPARSER_LOCAL_URL"] = omni_url

    from servers.vision.screenshot import capture_screenshot

    print("Waiting 5 seconds ...")
    time.sleep(5)
    print("Capturing screenshot (1920px JPEG) ...")
    t0 = time.time()
    b64, img_w, img_h, scale, backing_scale = capture_screenshot(
        target_width=1920, image_format="jpeg",
    )
    print(f"  {img_w}x{img_h}  scale={scale:.2f}  backing={backing_scale:.1f}  ({time.time() - t0:.2f}s)")

    import httpx

    print(f"Sending to OmniParser at {omni_url} ...")
    t0 = time.time()
    resp = httpx.post(
        f"{omni_url}/parse/",
        json={"base64_image": b64},
        timeout=600,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"  OmniParser responded ({time.time() - t0:.2f}s)")

    parsed = result.get("parsed_content_list", [])
    annotated_b64 = result.get("som_image_base64", b64)
    print(f"  {len(parsed)} elements detected")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("debug_logs", f"test_vision_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "screenshot_raw.png"), "wb") as f:
        f.write(base64.b64decode(b64))

    with open(os.path.join(out_dir, "screenshot_labeled.jpg"), "wb") as f:
        f.write(base64.b64decode(annotated_b64))

    lines = []
    for i, el in enumerate(parsed):
        content = el.get("content", "") or ""
        el_type = el.get("type", "unknown")
        label = "Icon" if "icon" in el_type.lower() else "Text"
        bbox = el.get("bbox", [])
        lines.append(f"[{i}] {label}: \"{content}\"  bbox={bbox}")

    with open(os.path.join(out_dir, "elements.txt"), "w") as f:
        f.write("\n".join(lines))

    with open(os.path.join(out_dir, "omniparser_raw.json"), "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  screenshot_raw.png      (what was sent)")
    print(f"  screenshot_labeled.jpg  (OmniParser annotated)")
    print(f"  elements.txt            ({len(parsed)} elements)")
    print(f"  omniparser_raw.json     (full OmniParser response)")


if __name__ == "__main__":
    main()
