"""CLI entry point for the mark desktop automation agent.

Spawns Vision and Action MCP servers as subprocesses,
connects the agent loop, and runs until the task completes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.loop import AgentLoop
from agent.mcp_client import connect_mcp
from config import MarkConfig


async def run_agent(task: str, config: MarkConfig) -> None:
    """Start MCP servers and run the agent loop."""
    async with connect_mcp("vision", sys.executable, ["-m", "servers.vision.server"]) as vision:
        async with connect_mcp("action", sys.executable, ["-m", "servers.action.server"]) as action:
            agent = AgentLoop(task, config, vision, action)
            result = await agent.run()
            print(f"\nResult: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mark -- macOS desktop automation agent",
    )
    parser.add_argument("task", help="Natural-language task to accomplish")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum agent steps (default: 100)",
    )
    parser.add_argument(
        "--screenshot-width", type=int, default=1280,
        help="Screenshot width in pixels (default: 1280)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = MarkConfig(
        model=args.model,
        max_steps=args.max_steps,
        screenshot_width=args.screenshot_width,
    )

    print(f"Task: {args.task}")
    print(f"Model: {config.model} | Max steps: {config.max_steps}")
    print("-" * 60)

    try:
        asyncio.run(run_agent(args.task, config))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
