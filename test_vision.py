"""Test observe(): call the Vision MCP function, save image + elements to debug/."""

import base64
import json
import os
import sys
import time

from servers.vision.server import observe

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")


def run():
    delay = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for i in range(delay, 0, -1):
        print(f"  {i}s …")
        time.sleep(1)

    result = json.loads(observe())
    run_id = int(time.time())
    os.makedirs(DEBUG_DIR, exist_ok=True)

    img_path = os.path.join(DEBUG_DIR, f"{run_id}_labeled.jpg")
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(result["image"]))

    elements_path = os.path.join(DEBUG_DIR, f"{run_id}_elements.txt")
    with open(elements_path, "w") as f:
        f.write(result["elements"])

    print(f"[{run_id}] Saved:")
    print(f"  Image:    {img_path}")
    print(f"  Elements: {elements_path}")
    print()
    print(result["elements"])


if __name__ == "__main__":
    run()
