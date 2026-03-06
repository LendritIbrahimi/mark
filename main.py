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
from agent.orchestrator import Orchestrator
from config import MarkConfig


async def run_agent(task: str, config: MarkConfig, *, use_orchestrator: bool = True) -> None:
    """Start MCP servers and run the agent loop."""
    async with connect_mcp("vision", sys.executable, ["-m", "servers.vision.server"]) as vision:
        async with connect_mcp("action", sys.executable, ["-m", "servers.action.server"]) as action:
            if use_orchestrator:
                runner = Orchestrator(task, config, vision, action)
            else:
                runner = AgentLoop(task, config, vision, action)
            result = await runner.run()
            print(f"\nResult: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mark -- macOS desktop automation agent",
    )
    parser.add_argument("task", help="Natural-language task to accomplish")
    parser.add_argument(
        "--model", default=None,
        help="LLM model name (default: o4-mini)",
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
        "--reasoning-effort", default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for o-series models (default: medium)",
    )
    parser.add_argument(
        "--no-orchestrator", action="store_true",
        help="Run as a single agent loop without goal decomposition",
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

    for noisy in ("httpx", "httpcore", "openai", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = MarkConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_steps=args.max_steps,
        screenshot_width=args.screenshot_width,
        send_images=not args.no_vision,
    )

    use_orchestrator = not args.no_orchestrator
    print(f"Task: {args.task}")
    if use_orchestrator:
        print("Mode: orchestrator (goal decomposition)")
    else:
        print("Mode: single agent loop")

    try:
        asyncio.run(run_agent(args.task, config, use_orchestrator=use_orchestrator))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
