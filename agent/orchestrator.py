"""Orchestrator -- decomposes a task into sub-goals and chains them."""

from __future__ import annotations

import json
import logging
import os

from pydantic import BaseModel, Field

from agent.debug import create_session_dir
from agent.llm import OpenAILLM
from agent.loop import AgentLoop
from agent.mcp_client import MCPClient
from agent.prompts import (
    build_decompose_messages,
    build_refine_goal_messages,
)
from config import MarkConfig

logger = logging.getLogger(__name__)


# -- Pydantic response models for orchestrator LLM calls --


class GoalDecomposition(BaseModel):
    goals: list[str] = Field(min_length=1)


class RefinedGoal(BaseModel):
    goal: str


class Orchestrator:
    """Splits a high-level task into sub-goals and executes each sequentially."""

    def __init__(
        self,
        task: str,
        config: MarkConfig,
        vision: MCPClient,
        action: MCPClient,
    ) -> None:
        self.task = task
        self.config = config
        self.vision = vision
        self.action = action

        planner_config = MarkConfig(
            model=config.orchestrator_model or config.model,
            temperature=config.temperature,
            llm_timeout=config.llm_timeout,
        )
        self.planner_llm = OpenAILLM(planner_config)
        self._session_dir = create_session_dir(task)

    async def run(self) -> str:
        """Decompose the task, execute each goal, and return the final result."""
        goals = await self._decompose()
        logger.info("Orchestrator: %d goals for task: %s", len(goals), self.task)
        for i, g in enumerate(goals, 1):
            logger.info("  Goal %d: %s", i, g)

        self._save_plan(goals)

        previous_result: str | None = None

        for idx, original_goal in enumerate(goals, 1):
            if previous_result is not None:
                goal = await self._refine_goal(original_goal, previous_result)
                logger.info("[Goal %d/%d] Refined: %s", idx, len(goals), goal)
            else:
                goal = original_goal
                logger.info("[Goal %d/%d] %s", idx, len(goals), goal)

            result, agent_dir = await self._execute_goal(goal)
            previous_result = result

            logger.info("[Goal %d/%d] Result: %s", idx, len(goals), previous_result)
            self._save_goal_result(idx, original_goal, goal, previous_result, agent_dir)

        return previous_result or "No result."

    # -- Phases --

    async def _decompose(self) -> list[str]:
        """Ask the LLM to break the task into ordered sub-goals."""
        messages = build_decompose_messages(self.task)
        response = await self.planner_llm.decide(messages, GoalDecomposition)
        goals = response.goals[: self.config.max_goals]
        return goals

    async def _refine_goal(self, original_goal: str, previous_result: str) -> str:
        """Rewrite *original_goal* to account for what was accomplished so far."""
        messages = build_refine_goal_messages(original_goal, previous_result)
        response = await self.planner_llm.decide(messages, RefinedGoal)
        return response.goal

    async def _execute_goal(self, goal: str) -> tuple[str, str]:
        """Run a fresh AgentLoop for a single goal and return (summary, session_dir)."""
        agent = AgentLoop(goal, self.config, self.vision, self.action)
        result = await agent.run()
        return result, agent.session_dir

    # -- Debug persistence --

    def _save_plan(self, goals: list[str]) -> None:
        path = os.path.join(self._session_dir, "plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"task": self.task, "goals": goals}, f, indent=2)

    def _save_goal_result(
        self,
        index: int,
        original_goal: str,
        refined_goal: str,
        result: str,
        agent_session_dir: str,
    ) -> None:
        goal_dir = os.path.join(self._session_dir, f"goal_{index:02d}")
        os.makedirs(goal_dir, exist_ok=True)
        path = os.path.join(goal_dir, "goal_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index": index,
                    "original_goal": original_goal,
                    "refined_goal": refined_goal,
                    "result": result,
                    "agent_session_dir": agent_session_dir,
                },
                f,
                indent=2,
            )
