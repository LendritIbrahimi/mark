"""Callback protocol for agent-to-UI communication."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AgentCallbacks:
    """Optional hooks that the UI (or any observer) can register.

    Every field defaults to ``None`` so the CLI path is unaffected.
    Callables are invoked from the agent's asyncio thread -- the UI
    is responsible for marshalling updates to its own thread
    (e.g. ``root.after()`` in Tkinter).
    """

    on_step_start: Callable[[int], Any] | None = None
    on_think: Callable[[int, str, str, list[dict]], Any] | None = None
    on_action_result: Callable[[int, str, dict], Any] | None = None
    on_goal_start: Callable[[int, int, str], Any] | None = None
    on_goal_end: Callable[[int, int, str], Any] | None = None
    on_decompose: Callable[[list[str]], Any] | None = None
    on_done: Callable[[str], Any] | None = None
    get_guidance: Callable[[], str | None] | None = None

    pause_event: threading.Event = field(
        default_factory=threading.Event,
    )
    plan_confirm_event: threading.Event = field(
        default_factory=threading.Event,
    )
    get_plan: Callable[[], list[str] | None] | None = None
    stop_requested: bool = False

    def __post_init__(self) -> None:
        self.pause_event.set()
        # plan_confirm_event starts clear (blocked) until user confirms

    def emit(self, name: str, *args: Any) -> None:
        cb = getattr(self, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception as exc:
                logger.error("Callback %s raised: %s", name, exc)
