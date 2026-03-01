"""CLI entry point for the mark desktop automation agent.

Spawns Vision and Action MCP servers as subprocesses,
connects the agent loop, and runs until the task completes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
        "--provider", default="openai",
        choices=["openai", "gemini", "ollama"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: provider-specific)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum agent steps (default: 100)",
    )
    parser.add_argument(
        "--screenshot-width", type=int, default=1280,
        help="Screenshot width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Don't send screenshots to the LLM (element list only)",
    )
    parser.add_argument(
        "--omniparser", action="store_true",
        help="Use OmniParser v2 for element detection instead of accessibility tree",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed timing and debug information",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        os.environ["MARK_LOG_LEVEL"] = "DEBUG"
    else:
        from agent.console import ConsoleFormatter
        handler = logging.StreamHandler()
        handler.setFormatter(ConsoleFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])
        os.environ["MARK_LOG_LEVEL"] = "WARNING"

    for noisy in ("httpx", "httpcore", "openai", "google", "google_genai", "ollama", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    default_models = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "ollama": "llama3.2-vision",
    }
    model = args.model or default_models.get(args.provider, "gpt-4o-mini")

    config = MarkConfig(
        provider=args.provider,
        model=model,
        max_steps=args.max_steps,
        screenshot_width=args.screenshot_width,
        send_images=not args.no_vision,
        use_omniparser=args.omniparser,
    )

    print(f"Task: {args.task}")

    try:
        asyncio.run(run_agent(args.task, config))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
