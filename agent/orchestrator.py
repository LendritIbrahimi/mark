"""Orchestrator -- decomposes a task into sub-goals."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field

from agent.callbacks import AgentCallbacks
from agent.debug import create_session_dir
from agent.llm import OpenAILLM
from agent.loop import AgentLoop
from agent.mcp_client import MCPClient
from agent.prompts import (
    build_decompose_messages,
    build_refine_goal_messages,
    build_replan_messages,
    build_triage_messages,
    build_validate_goal_messages,
)
from agent.config import MarkConfig

logger = logging.getLogger(__name__)


class TaskTriage(BaseModel):
    complexity: Literal["simple", "complex"]


class GoalDecomposition(BaseModel):
    goals: list[str] = Field(min_length=1)


class RefinedGoal(BaseModel):
    goal: str


class GoalValidation(BaseModel):
    success: bool
    reason: str


class Orchestrator:
    """Splits a task into sub-goals and executes each."""

    def __init__(
            self,
            task: str,
            config: MarkConfig,
            vision: MCPClient,
            action: MCPClient,
            callbacks: AgentCallbacks | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.vision = vision
        self.action = action
        self._cb = callbacks or AgentCallbacks()

        planner_config = MarkConfig(
            model=(
                    config.orchestrator_model or config.model
            ),
            temperature=config.temperature,
            llm_timeout=config.llm_timeout,
        )
        self.planner_llm = OpenAILLM(planner_config)

    async def run(self) -> str:
        session_dir = (
            create_session_dir(self.task) if self.config.save_debug_logs else ""
        )
        is_simple = await self._triage()

        if is_simple:
            goals = [self.task]
            logger.info(
                "Simple task, skipping decomposition.",
            )
        else:
            goals = await self._decompose()
            self._cb.emit("on_decompose", goals)
            if self.config.allow_plan_edit:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._cb.plan_confirm_event.wait,
                )
                if self._cb.get_plan is not None:
                    modified = self._cb.get_plan()
                    if modified:
                        goals = modified
            logger.info(
                "Decomposed into %d goals.", len(goals),
            )
            for i, g in enumerate(goals, 1):
                logger.info("  Goal %d: %s", i, g)

        previous_result: str | None = None
        replans_done = 0
        goals_log: list[str] = []

        while True:
            failed_in_plan = 0

            for idx, original_goal in enumerate(goals, 1):
                if previous_result is not None:
                    goal = await self._refine_goal(
                        original_goal, previous_result,
                    )
                    logger.info(
                        "[Goal %d/%d] Refined: %s",
                        idx, len(goals), goal,
                    )
                else:
                    goal = original_goal
                    logger.info(
                        "[Goal %d/%d] %s",
                        idx, len(goals), goal,
                    )

                self._cb.emit(
                    "on_goal_start", idx, len(goals), goal,
                )
                result = await self._execute_goal(goal, idx, session_dir)

                goal_succeeded = False
                for attempt in range(
                        self.config.max_goal_retries,
                ):
                    validation = await self._validate_goal(
                        goal, result,
                    )
                    if validation.success:
                        goal_succeeded = True
                        break
                    logger.warning(
                        "[Goal %d/%d] Failed (attempt %d): %s",
                        idx,
                        len(goals),
                        attempt + 1,
                        validation.reason,
                    )
                    goal = await self._refine_goal(
                        goal,
                        (
                            f"FAILED: {validation.reason}. "
                            f"{result}"
                        ),
                    )
                    logger.info(
                        "[Goal %d/%d] Retry: %s",
                        idx, len(goals), goal,
                    )
                    self._cb.emit("on_goal_start", idx, len(goals), goal)
                    result = await self._execute_goal(goal, idx, session_dir)

                previous_result = result
                self._cb.emit(
                    "on_goal_end",
                    idx, len(goals), previous_result,
                )
                logger.info(
                    "[Goal %d/%d] Result: %s",
                    idx, len(goals), previous_result,
                )

                if not goal_succeeded:
                    failed_in_plan += 1
                    goals_log.append(
                        f"FAILED — {original_goal}: {result}",
                    )
                    if (
                        failed_in_plan >= self.config.max_plan_failures
                        and replans_done < self.config.max_replans
                    ):
                        replans_done += 1
                        logger.warning(
                            "Plan failing, triggering replan (%d/%d).",
                            replans_done,
                            self.config.max_replans,
                        )
                        goals = await self._replan(
                            goals_log, previous_result,
                        )
                        self._cb.emit("on_decompose", goals)
                        if self.config.allow_plan_edit:
                            self._cb.plan_confirm_event.clear()
                            await asyncio.get_running_loop().run_in_executor(
                                None, self._cb.plan_confirm_event.wait,
                            )
                            if self._cb.get_plan is not None:
                                modified = self._cb.get_plan()
                                if modified:
                                    goals = modified
                        break  # restart while loop with new goals
                else:
                    goals_log.append(f"OK — {original_goal}")
            else:
                break  # for loop completed normally — exit while

        return previous_result or "No result."

    async def _triage(self) -> bool:
        messages = build_triage_messages(self.task)
        try:
            response = await self.planner_llm.decide(
                messages, TaskTriage,
            )
            return response.complexity == "simple"
        except Exception as exc:
            logger.error(
                "Triage failed (%s), "
                "falling back to decomposition.",
                exc,
            )
            return False

    async def _decompose(self) -> list[str]:
        messages = build_decompose_messages(self.task)
        try:
            response = await self.planner_llm.decide(
                messages, GoalDecomposition,
            )
            return response.goals[: self.config.max_goals]
        except Exception as exc:
            logger.error(
                "Decompose failed (%s), treating as single goal.", exc,
            )
            return [self.task]

    async def _refine_goal(
            self, original_goal: str, previous_result: str,
    ) -> str:
        messages = build_refine_goal_messages(
            original_goal, previous_result,
        )
        try:
            response = await self.planner_llm.decide(
                messages, RefinedGoal,
            )
            return response.goal
        except Exception as exc:
            logger.error(
                "Goal refinement failed (%s), using original.", exc,
            )
            return original_goal

    async def _validate_goal(
            self, goal: str, result: str,
    ) -> GoalValidation:
        messages = build_validate_goal_messages(
            self.task, goal, result,
        )
        try:
            return await self.planner_llm.decide(
                messages, GoalValidation,
            )
        except Exception:
            return GoalValidation(
                success=True, reason="validation skipped",
            )

    async def _replan(
            self,
            goals_log: list[str],
            current_state: str | None,
    ) -> list[str]:
        messages = build_replan_messages(
            self.task, goals_log, current_state,
        )
        try:
            response = await self.planner_llm.decide(
                messages, GoalDecomposition,
            )
            return response.goals[: self.config.max_goals]
        except Exception as exc:
            logger.error(
                "Replan failed (%s), using original task.", exc,
            )
            return [self.task]

    async def _execute_goal(
            self, goal: str, goal_idx: int = 1, session_dir: str = "",
    ) -> str:
        agent = AgentLoop(
            goal, self.config, self.vision, self.action,
            callbacks=self._cb,
            goal_idx=goal_idx,
            session_dir=session_dir or None,
        )
        return await agent.run()
