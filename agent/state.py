"""State manager -- holds all context the LLM needs across steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StateManager:
    """Mutable state for the agent loop."""

    goal: str
    max_steps: int = 100
    max_recent_results: int = 5

    step: int = 0
    element_positions: dict[int, dict] = field(default_factory=dict)
    recent_results: list[str] = field(default_factory=list)
    image_b64: str = ""
    elements: str = ""
    scale: float = 1.0
    backing_scale: float = 2.0
    loop_warning: str = ""

    # Anti-loop tracking
    _recent_failed: list[str] = field(default_factory=list)
    _recent_all: list[str] = field(default_factory=list)
    _prev_element_count: int = -1
    _consecutive_empty: int = 0

    def update_ui(self, perception: dict) -> None:
        """Update the UI state from a Vision MCP ``observe()`` response."""
        self.image_b64 = perception.get("image", "")
        self.elements = perception.get("elements", "")
        self.scale = perception.get("scale", 1.0)
        self.backing_scale = perception.get("backing_scale", 2.0)

        positions = perception.get("element_positions", [])
        self.element_positions = {p["id"]: p for p in positions}

        current_count = len(positions)
        if self._prev_element_count == current_count and self._prev_element_count >= 0:
            if self._recent_all:
                self.loop_warning = (
                    "Screen appears unchanged after your last action. "
                    "Try a different approach."
                )
                logger.debug("Screen unchanged warning (elements=%d)", current_count)
        else:
            self.loop_warning = ""
        self._prev_element_count = current_count

        logger.debug("UI updated: %d elements", len(positions))

    def resolve_element(self, element_id: int) -> tuple[int, int] | None:
        """Resolve an element ID to its center (x, y) in pixel coordinates."""
        pos = self.element_positions.get(element_id)
        if pos is None:
            return None
        return pos["x"], pos["y"]

    def record_result(self, text: str) -> None:
        """Add an action result to the rolling window."""
        self.recent_results.append(text)
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results = self.recent_results[-self.max_recent_results:]

    def track_action(self, action_key: str, failed: bool) -> None:
        """Track actions for anti-loop detection."""
        if failed:
            self._recent_failed.append(action_key)
            if len(self._recent_failed) > 10:
                self._recent_failed = self._recent_failed[-10:]
            count = self._recent_failed.count(action_key)
            if count >= 2:
                self.loop_warning = (
                    f"Action '{action_key}' has failed {count} times. "
                    f"Try a completely different action."
                )
        else:
            self._recent_all.append(action_key)
            if len(self._recent_all) > 12:
                self._recent_all = self._recent_all[-12:]
            count = self._recent_all.count(action_key)
            if count >= 3:
                self.loop_warning = (
                    f"Action '{action_key}' repeated {count} times with no progress. "
                    f"Try a completely different approach."
                )
                self._recent_all.clear()

    def record_empty_response(self, *, success: bool) -> None:
        """Track consecutive empty/failed LLM responses for escalating warnings."""
        if success:
            self._consecutive_empty = 0
            return
        self._consecutive_empty += 1
        if self._consecutive_empty >= 3:
            self.loop_warning = f"No response {self._consecutive_empty} times in a row."
            logger.warning("Consecutive empty LLM responses: %d", self._consecutive_empty)
