"""CLI entry point for the mark desktop automation agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from agent.loop import AgentLoop
from agent.mcp_client import connect_mcp
from agent.orchestrator import Orchestrator
from agent.config import MarkConfig


async def run_agent(
        task: str,
        config: MarkConfig,
        *,
        use_orchestrator: bool = True,
) -> None:
    async with connect_mcp(
            "vision", sys.executable, ["-m", "servers.vision.server"],
    ) as vision:
        async with connect_mcp(
                "action", sys.executable, ["-m", "servers.action.server"],
        ) as action:
            if use_orchestrator:
                runner = Orchestrator(
                    task, config, vision, action,
                )
            else:
                runner = AgentLoop(
                    task, config, vision, action,
                )
            result = await runner.run()
            print(f"\nResult: {result}")


async def _run(
        task: str,
        config: MarkConfig,
        *,
        use_orchestrator: bool = True,
) -> None:
    """Optionally set OmniParser URL, then run the agent."""
    if config.use_omniparser:
        url = f"http://127.0.0.1:{config.omniparser_port}"
        os.environ["OMNIPARSER_LOCAL_URL"] = url
        print(f"OmniParser: expecting server at {url}")
    await run_agent(
        task, config, use_orchestrator=use_orchestrator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mark -- macOS desktop automation agent",
    )
    parser.add_argument(
        "task", help="Natural-language task to accomplish",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model name (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Don't send screenshots to the LLM",
    )
    parser.add_argument(
        "--reasoning-effort", default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for o-series models",
    )
    parser.add_argument(
        "--no-orchestrator", action="store_true",
        help="Run without goal decomposition",
    )
    parser.add_argument(
        "--omniparser", action="store_true",
        help="Use OmniParser for UI element detection (starts local server)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
    )
    for name in ("httpx", "httpcore", "openai", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)

    config = MarkConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        send_images=not args.no_vision,
        use_omniparser=args.omniparser,
    )

    use_orchestrator = not args.no_orchestrator
    print(f"Task: {args.task}")
    mode = "orchestrator" if use_orchestrator else "single"
    print(f"Mode: {mode}")

    for i in range(5, 0, -1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)
    print("Go!\n")

    try:
        asyncio.run(
            _run(
                args.task, config,
                use_orchestrator=use_orchestrator,
            ),
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
