"""State manager for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateManager:
    """Mutable state for the agent loop."""

    goal: str
    max_recent_results: int = 5

    step: int = 0
    element_positions: dict[int, dict] = field(
        default_factory=dict,
    )
    recent_results: list[str] = field(
        default_factory=list,
    )
    all_results: list[str] = field(default_factory=list)
    image_b64: str = ""
    elements: str = ""
    scale: float = 1.0
    backing_scale: float = 2.0
    loop_warning: str = ""
    user_guidance: str = ""

    _recent_failed: list[str] = field(
        default_factory=list,
    )
    _recent_all: list[str] = field(
        default_factory=list,
    )
    _prev_element_count: int = -1
    _prev_elements_hash: int = -1
    _significant_ui_change: bool = False
    _consecutive_empty: int = 0
    _consecutive_stale_steps: int = 0

    def update_ui(self, perception: dict) -> None:
        self.image_b64 = perception.get("image", "")
        self.elements = perception.get("elements", "")
        self.scale = perception.get("scale", 1.0)
        self.backing_scale = perception.get(
            "backing_scale", 2.0,
        )

        positions = perception.get(
            "element_positions", [],
        )
        self.element_positions = {
            p["id"]: p for p in positions
        }

        elements_text = perception.get("elements", "")
        current_count = len(positions)
        elements_hash = hash(elements_text)

        count_same = (
            self._prev_element_count == current_count
            and self._prev_element_count >= 0
        )
        content_same = (
            self._prev_elements_hash == elements_hash
            and self._prev_elements_hash != -1
        )

        if count_same and content_same:
            if self._recent_all:
                self.loop_warning = (
                    "Screen appears unchanged after "
                    "your last action. "
                    "Try a different approach."
                )
            self._significant_ui_change = False
        else:
            self.loop_warning = ""
            if self._prev_element_count >= 0:
                delta = abs(current_count - self._prev_element_count)
                self._significant_ui_change = (
                    delta > max(5, self._prev_element_count // 4)
                    or not content_same
                )
            else:
                self._significant_ui_change = True

        self._prev_element_count = current_count
        self._prev_elements_hash = elements_hash

    def resolve_element(
            self, element_id: int,
    ) -> tuple[int, int] | None:
        pos = self.element_positions.get(element_id)
        if pos is None:
            return None
        return pos["x"], pos["y"]

    def record_result(self, text: str) -> None:
        self.all_results.append(text)
        self.recent_results.append(text)
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results = (
                self.recent_results[
                    -self.max_recent_results:
                ]
            )

    def track_action(
            self, action_key: str, failed: bool,
    ) -> None:
        if failed:
            self._recent_failed.append(action_key)
            if len(self._recent_failed) > 10:
                self._recent_failed = (
                    self._recent_failed[-10:]
                )
            count = self._recent_failed.count(action_key)
            if count >= 2:
                self.loop_warning = (
                    f"Action '{action_key}' has failed "
                    f"{count} times. "
                    f"Try a completely different action."
                )
            action_name = action_key.split("(")[0]
            name_count = sum(
                1 for k in self._recent_failed
                if k.split("(")[0] == action_name
            )
            if name_count >= 3:
                self.loop_warning = (
                    f"'{action_name}' has failed {name_count} times. "
                    f"Switch approach entirely: use keyboard shortcuts "
                    f"(Tab, Enter, Cmd+F), try scrolling, or interact "
                    f"with a different element."
                )
                self._recent_failed.clear()
        else:
            self._recent_all.append(action_key)
            if len(self._recent_all) > 12:
                self._recent_all = (
                    self._recent_all[-12:]
                )
            count = self._recent_all.count(action_key)
            if count >= 3:
                self.loop_warning = (
                    f"Action '{action_key}' repeated "
                    f"{count} times with no progress. "
                    f"Try a completely different approach."
                )
                self._recent_all.clear()

    def record_empty_response(
            self, *, success: bool,
    ) -> None:
        if success:
            self._consecutive_empty = 0
            return
        self._consecutive_empty += 1
        if self._consecutive_empty >= 3:
            self.loop_warning = (
                f"No response "
                f"{self._consecutive_empty} "
                f"times in a row."
            )

    def check_stale(self, max_stale: int) -> bool:
        if self.loop_warning:
            self._consecutive_stale_steps += 1
        elif self._significant_ui_change:
            self._consecutive_stale_steps = 0
        return self._consecutive_stale_steps >= max_stale
