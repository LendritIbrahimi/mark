"""Centralized configuration for the mark desktop automation agent."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "gpt-5-nano"

REASONING_PREFIXES = ("o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    if model.startswith("gpt-5"):
        return True
    base = model.split("-")[0] if "-" in model else model
    return base in REASONING_PREFIXES


@dataclass
class MarkConfig:
    model: str | None = None
    temperature: float = 0.1
    reasoning_effort: str = "low"
    llm_timeout: float = 180.0
    llm_max_retries: int = 2
    max_actions_per_step: int = 3
    max_failures: int = 5
    max_stale_steps: int = 5
    step_delay: float = 5
    initial_delay: float = 0.0
    send_images: bool = True
    post_action_delay: float = 1.0
    max_messages: int = 20
    max_recent_results: int = 5
    max_goals: int = 5
    max_goal_retries: int = 2
    max_plan_failures: int = 2
    max_replans: int = 1
    orchestrator_model: str | None = None
    mcp_timeout: float = 600.0
    allow_plan_edit: bool = True
    save_debug_logs: bool = True
    use_omniparser: bool = False
    omniparser_port: int = 8100
