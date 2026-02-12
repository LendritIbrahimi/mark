"""Prompt templates for the macOS automation agent.

All prompt construction lives here -- the agent and LLM client
only deal with pre-built message lists.
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

ACTION_DOCS = """\
Mouse actions (always use element_id):
- click(element_id) -- Left click
- double_click(element_id) -- Double click
- right_click(element_id) -- Right click (context menu)
- hover(element_id) -- Move mouse without clicking
- drag(from_element_id, to_element_id) -- Drag
- scroll(direction, amount) -- Scroll at cursor. direction: up/down/left/right, amount: 1-10

Keyboard actions:
- type_text(text, element_id?, submit?) -- Type text. If element_id given, clicks it first. submit=true presses Enter.
- press_key(key) -- Single key: enter, tab, escape, space, delete, up, down, left, right, f1-f12, etc.
- hotkey(keys) -- Shortcut: ["command", "c"], ["command", "v"], ["command", "shift", "s"], etc.

Agent actions:
- wait(seconds, reason?) -- Wait 0.5-10 seconds for loading
- done(text) -- Mark task complete with a result description"""


def build_system_prompt() -> dict:
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
2. An element list mapping each number to a label: [5] "Google Search" (AXButton)

ALWAYS use element IDs for actions: click(element_id=5), type_text(text="hello", element_id=5).
Never use raw x,y coordinates. If the element you need is not listed, try scrolling, waiting, or a different approach.

## Response Format (JSON)

{RESPONSE_SCHEMA}

## Available Actions

{ACTION_DOCS}

## Rules

1. LOOK at the screenshot carefully before acting.
2. After acting, check the next screenshot to verify it worked.
3. Never repeat a failed action -- try a different approach.
4. Only call done() when you see visual proof the task is complete."""

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
