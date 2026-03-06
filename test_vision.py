"""Quick test: wait 5s, take an MCP screenshot, save to debug_logs/."""

import asyncio
import base64
import json
import os
import sys

from agent.mcp_client import connect_mcp


async def main() -> None:
    print("Waiting 5 seconds...")
    await asyncio.sleep(5)

    async with connect_mcp("vision", sys.executable, ["-m", "servers.vision.server"]) as vision:
        print("Taking screenshot...")
        result = await vision.call_tool("observe", {
            "width": 1280,
            "max_elements": 150,
        }, timeout=30.0)

    out_dir = os.path.join("debug_logs", "test_vision")
    os.makedirs(out_dir, exist_ok=True)

    if result.get("image"):
        img_path = os.path.join(out_dir, "screenshot_labeled.jpg")
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(result["image"]))
        print(f"Saved: {img_path}")

    elements = result.get("elements", "")
    elem_path = os.path.join(out_dir, "elements.txt")
    with open(elem_path, "w", encoding="utf-8") as f:
        f.write(elements)
    print(f"Saved: {elem_path} ({elements.count(chr(10)) + 1} elements)")

    positions = result.get("element_positions", [])
    pos_path = os.path.join(out_dir, "positions.json")
    with open(pos_path, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)
    print(f"Saved: {pos_path}")


if __name__ == "__main__":
    asyncio.run(main())
