"""Centralized configuration for the mark desktop automation agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarkConfig:
    """All tuneable settings for mark."""

    # -- LLM --
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # -- Vision --
    screenshot_width: int = 1280
    max_elements: int = 150

    # -- Agent loop --
    max_steps: int = 100
    max_actions_per_step: int = 10
    max_failures: int = 5
    step_delay: float = 1.0
    initial_delay: float = 5.0
    post_action_delay: float = 2.0

    # -- Conversation history --
    max_messages: int = 20
    max_recent_results: int = 5

    # -- MCP --
    mcp_timeout: float = 30.0
