"""Prompt templates for the macOS automation agent.

All prompt construction lives here -- the agent loop
only deals with pre-built message dicts.
"""

from __future__ import annotations

import platform
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import StateManager

RESPONSE_SCHEMA = """\
{
  "thought": "<What you see on screen and what to do next>",
  "actions": [
    {"name": "<action_name>", "params": {<params>}}
  ]
}"""


def build_system_prompt(action_docs: str = "") -> dict:
    """Build the one-time system message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    os_ver = platform.mac_ver()[0]
    os_label = f"macOS {os_ver}" if os_ver else platform.platform()

    content = f"""\
You are a desktop automation agent controlling a real macOS computer.
OS: {os_label} | Date: {now}

## How You See the Screen

Each step you receive:
1. A SCREENSHOT with numbered bounding boxes on detected UI elements -- your primary visual input.
2. An ELEMENT LIST mapping each number to a label, for example:
   [5] Button: "Submit"
   [12] TextBox: "Search"

ALWAYS use element IDs for actions: click(element_id=5), type_text(text="hello", element_id=12).
Never use raw x,y coordinates. If the element you need is not listed, try scrolling, waiting, or a different approach.

## Response Format (JSON)

{RESPONSE_SCHEMA}

## Available Actions

{action_docs}

## Rules

1. LOOK at the screenshot carefully before acting. Match numbered boxes with the element list.
2. After acting, check the next screenshot to verify it worked.
3. Never repeat a failed action -- try a completely different approach.
4. Only call done() when you see visual proof the task is complete.
5. You MUST always respond with valid JSON. If unsure, use wait(seconds=1)."""

    return {"role": "system", "content": content}


def build_step_message(state: StateManager) -> dict:
    """Build the per-step user message from current state."""
    parts: list[str] = []

    parts.append(f"Task: {state.goal}")
    parts.append(f"Step {state.step}/{state.max_steps}")

    if state.loop_warning:
        parts.append(f"\nWARNING: {state.loop_warning}")

    recent = state.recent_results
    if recent:
        failures = [r for r in recent if "FAILED" in r or "ERROR" in r]
        successes = [r for r in recent if r not in failures]
        if failures:
            parts.append("\nFailed (try differently):")
            for r in failures:
                parts.append(f"  - {r}")
        if successes:
            parts.append("\nPrevious results:")
            for r in successes:
                parts.append(f"  - {r}")

    if state.elements:
        parts.append(f"\nScreen elements:\n{state.elements}")
    else:
        parts.append("\nNo elements detected on screen.")

    return {"role": "user", "content": "\n".join(parts)}


# -- Orchestrator prompts --


def build_decompose_messages(task: str) -> list[dict]:
    """Build messages that ask the LLM to split *task* into ordered sub-goals."""
    system = {
        "role": "system",
        "content": (
            "You are a planning assistant for a macOS desktop automation agent.\n"
            "Given a high-level task, decompose it into an ordered list of "
            "self-contained goals that can be executed one after another.\n\n"
            "Rules:\n"
            "1. Each goal must be a concrete, actionable instruction.\n"
            "2. Goals run sequentially -- later goals can assume earlier ones succeeded.\n"
            "3. Keep the list short (2-6 goals). Don't over-split trivial steps.\n"
            "4. The final goal should produce the end result the user asked for.\n\n"
            'Respond with JSON: {"goals": ["goal 1", "goal 2", ...]}'
        ),
    }
    user = {"role": "user", "content": f"Task: {task}"}
    return [system, user]


def build_refine_goal_messages(original_goal: str, previous_result: str) -> list[dict]:
    """Build messages that refine the next goal using the previous goal's result."""
    system = {
        "role": "system",
        "content": (
            "You are a planning assistant for a macOS desktop automation agent.\n"
            "You will receive the next planned goal and a summary of the current "
            "system state (what was accomplished so far). Your job is to blend them "
            "into a single concrete, actionable instruction that a desktop automation "
            "agent can execute immediately.\n\n"
            "Rules:\n"
            "1. Replace vague references with specifics from the current state "
            "(e.g. which app is open, what URL is loaded, what was copied).\n"
            "2. If the current state already satisfies part of the goal, skip that part.\n"
            "3. Keep it as a single short instruction -- do NOT split into sub-steps.\n"
            "4. The refined goal must be self-contained: an agent reading ONLY this "
            "goal (with no prior context) should know exactly what to do.\n\n"
            "Example:\n"
            '  Current state: "Copied a funny dog meme image, closed the tab, Safari is still open."\n'
            '  Next planned goal: "Open Messenger and send John the funny image"\n'
            '  Refined: "Open messenger.com in Safari and paste the copied image into the chat with John"\n\n'
            'Respond with JSON: {"goal": "<refined goal>"}'
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Current state: {previous_result}\n\n"
            f"Next planned goal: {original_goal}"
        ),
    }
    return [system, user]


def build_goal_self_summary_messages(goal: str, recent_results: list[str]) -> list[dict]:
    """Build messages for the AgentLoop to summarize its own execution.

    Called at the end of a goal's perceive-think-act loop so the goal
    produces a self-contained ``{result: str}`` output.
    """
    results_text = "\n".join(f"- {r}" for r in recent_results) if recent_results else "(no actions taken)"
    system = {
        "role": "system",
        "content": (
            "You are a summarization assistant for a macOS desktop automation agent.\n"
            "Given a goal and the sequence of action results from executing it, "
            "produce a brief summary (1-3 sentences) of what was accomplished "
            "and the current state of the screen / system.\n"
            "Focus on the outcome and what is visible or ready for the next step.\n\n"
            'Respond with JSON: {"result": "<summary>"}'
        ),
    }
    user = {
        "role": "user",
        "content": f"Goal: {goal}\n\nAction results:\n{results_text}",
    }
    return [system, user]
