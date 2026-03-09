"""Centralized configuration for the mark desktop automation agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarkConfig:
    """All tuneable settings for mark."""

    # -- LLM --
    model: str | None = None
    temperature: float = 0.2
    reasoning_effort: str = "medium"
    llm_timeout: float = 30.0

    # -- Vision --
    screenshot_width: int = 1280
    max_elements: int = 150

    # -- Agent loop --
    max_steps: int = 100
    max_actions_per_step: int = 10
    max_failures: int = 5
    max_stale_steps: int = 4
    step_delay: float = 1.0
    initial_delay: float = 0.0
    send_images: bool = True
    post_action_delay: float = 2.0

    # -- Conversation history --
    max_messages: int = 20
    max_recent_results: int = 5

    # -- Orchestrator --
    max_goals: int = 6
    max_goal_retries: int = 1
    orchestrator_model: str | None = None

    # -- MCP --
    mcp_timeout: float = 30.0
