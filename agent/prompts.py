"""Prompt templates for the macOS automation agent."""

from __future__ import annotations

import platform
import textwrap
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import StateManager

RESPONSE_SCHEMA = """\
{
  "observation": "<What you see on the screen>",
  "thought": "<What to do next and why>",
  "actions": [
    {"name": "<action_name>", "params": {<params>}}
  ]
}"""


def build_system_prompt(action_docs: str = "") -> dict:
    """Build the system prompt for the agent."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    os_ver = platform.mac_ver()[0]
    os_label = f"macOS {os_ver}" if os_ver else platform.platform()

    content = textwrap.dedent(f"""\
        You are a desktop automation agent controlling a real macOS computer.
        OS: {os_label} | Date: {now}

        ## How You See the Screen

        Each step you receive:
        1. A SCREENSHOT with numbered bounding boxes.
           This is your PRIMARY input.
        2. An ELEMENT LIST mapping each number to a label
           (e.g. [5] Button: "Submit").

        Use element IDs for actions:
          click(element_id=5), type_text(text="hello", element_id=12).
        Never use raw x,y coordinates.
        Elements tagged [OFFSCREEN] are scrolled out of view -- scroll to reveal them first.
        Elements tagged [HIDDEN] have no bounding box on the screenshot -- do not interact with them.
        If the element you need is not listed, try scrolling or waiting.

        ## Response Format (JSON)

        {RESPONSE_SCHEMA}

        IMPORTANT: Always fill in "observation" first.
        Describe what you see BEFORE reasoning about actions.

        ## Available Actions

        {action_docs}

        ## Rules

        1. OBSERVE the screenshot first. Describe what you see BEFORE deciding.
        2. After acting, check the next screenshot to confirm the action worked.
        3. Never repeat a failed action with the same parameters.
        4. Complete the goal AS LITERALLY STATED.
        5. Do not infer extra steps beyond what was asked.
        6. BEFORE acting, check if the goal is already met — if so, call done().
        7. Always respond with valid JSON.
        8. Use wait() generously. Real UIs are slow. Always add a wait() after:
           - Typing into a URL/search bar BEFORE pressing Enter
           - Pressing Enter to navigate or submit
           - Opening an app or switching windows
           - Any action that triggers a page load or animation
           When in doubt, add a wait(2). It is always better to wait than to
           act on a stale screen.

        ## When Stuck or Failing

        If an action fails or the screen does not change:
        - Try keyboard alternatives: Tab (move focus), Enter (activate),
          Escape (dismiss dialogs), Cmd+F (search), arrow keys for navigation.
        - Try scrolling (up/down/left/right) to reveal hidden elements.
        - Use right_click to open a context menu with more options.
        - Use wait() (1-3 seconds) for slow UI transitions before acting.
        - If an element is not in the list, try a different navigation path:
          use the menu bar, a keyboard shortcut, or Spotlight (Cmd+Space).
        - If the same action type keeps failing, abandon it entirely and
          describe a new strategy in your thought.
        - If the goal is truly impossible (app missing, permission denied,
          element never appears), call done() explaining what was tried.
    """)

    return {"role": "system", "content": content}


def build_step_message(state: StateManager) -> dict:
    """Build the user message for one agent step."""
    parts: list[str] = [
        f"Task: {state.goal}",
        f"Step {state.step}",
    ]

    if state.loop_warning:
        parts.append(f"\nWARNING: {state.loop_warning}")

    recent = state.recent_results
    if recent:
        failures = [r for r in recent if "FAILED" in r or "ERROR" in r]
        successes = [r for r in recent if r not in failures]
        if failures:
            parts.append("\nFailed (try differently):")
            parts.extend(f"  - {r}" for r in failures)
        if successes:
            parts.append("\nPrevious results:")
            parts.extend(f"  - {r}" for r in successes)

    if state.user_guidance:
        parts.append(f"\nUser guidance: {state.user_guidance}")
        state.user_guidance = ""

    if state.elements:
        parts.append(f"\nScreen elements:\n{state.elements}")
    else:
        parts.append("\nNo elements detected on screen.")

    return {"role": "user", "content": "\n".join(parts)}


def build_decompose_messages(task: str) -> list[dict]:
    """Build messages for task decomposition."""
    system_content = textwrap.dedent("""\
        You are a planning assistant for a macOS desktop automation agent.
        Given a high-level task, decompose it into a MINIMAL list of goals.

        ## Agent capabilities

        The agent that executes each goal is highly capable. Each step it:
        - Captures a fresh screenshot with labeled UI elements
        - Reasons about what it sees and what to do next
        - Can click, type, scroll, press keys, use hotkeys, drag, wait
        - Handles multi-step interactions within a single goal automatically

        Because the agent is so capable, you should keep goals BROAD.
        A single goal can involve many clicks, typing, navigating, scrolling,
        and waiting -- the agent handles all of that on its own.

        ## Rules

        1. MINIMIZE the number of goals. Most tasks need only 1-3 goals.
           The agent can handle complex multi-step interactions within one goal.
           Only split when there is a HARD boundary: switching between different
           apps, or when one stage must be fully verified before the next begins.
        2. Every goal MUST end with a clear, visually verifiable completion
           condition so the agent knows when to call done().
           Bad:  "Search for black dog images"
           Good: "Open Safari, search Google for 'black dog', and click the
                  Images tab — the image grid should be visible"
        3. Each goal must be concrete: name the app/website explicitly, include
           exact URLs or labels when possible.
        4. If an app might already be open, write "Open or switch to X".
        5. Avoid open-ended goals like "find" or "look for" without stating
           what counts as success.
        6. Never add goals for agent internals (screenshots, element detection).

        ## Examples

        Task: "hover the mouse on the first pug dog you see"
          -> {"goals": [
               "Hover the mouse over the first pug dog image visible on screen — the cursor should be resting on a pug image"
             ]}

        Task: "open Safari and search for cute cats"
          -> {"goals": [
               "Open Safari (or switch to it), search Google for 'cute cats' — the search results page should be visible"
             ]}

        Task: "download an image of a black dog"
          -> {"goals": [
               "Open Safari (or switch to it), search Google for 'black dog', and go to Google Images — the image grid should be visible",
               "Click on a black dog image, then save it (right-click > Save Image) — the download should start or the save dialog should appear"
             ]}

        Task: "send John a message on Messenger saying hello"
          -> {"goals": [
               "Open Messenger (messenger.com in Safari or the app), find and open the conversation with John — John's chat thread should be visible",
               "Type 'hello' and send the message — the sent message should appear in the conversation"
             ]}

        Task: "create a note in Notes titled 'Shopping list' with items milk, eggs, bread"
          -> {"goals": [
               "Open the Notes app (or switch to it), create a new note, type 'Shopping list' as the title then add milk, eggs, bread on separate lines — the note with all items should be visible"
             ]}

        Respond with JSON: {"goals": ["goal 1", ...]}
    """)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Task: {task}"},
    ]


def build_triage_messages(task: str) -> list[dict]:
    """Build messages for task complexity triage."""
    system_content = textwrap.dedent("""\
        You classify tasks for a macOS desktop automation agent.
        The agent can see the screen (screenshot + UI elements) and perform
        mouse/keyboard actions each step. It is highly capable and handles
        multi-step interactions automatically within a single goal.

        Classify the task as:
        - "simple" -- can be accomplished without switching between different
          apps or distinct workflow phases. This includes multi-step tasks
          within a single app (e.g. open Safari and search for something,
          click several buttons, fill out a form, navigate a website).
        - "complex" -- requires switching between different apps, or involves
          clearly distinct phases that must be verified independently
          (e.g. copy something from app A and paste in app B, download a file
          then attach it somewhere, multi-app workflows).

        When in doubt, classify as "simple". The agent is capable enough to
        handle most tasks in a single goal.

        Respond with JSON: {"complexity": "simple"} or {"complexity": "complex"}
    """)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Task: {task}"},
    ]


def build_refine_goal_messages(
    original_goal: str, previous_result: str,
) -> list[dict]:
    """Build messages for refining a goal with context."""
    system_content = textwrap.dedent("""\
        You are a planning assistant for a macOS desktop automation agent.
        You will receive the next planned goal and a summary of the current
        system state (what was accomplished so far). Your job is to blend
        them into a single concrete, actionable instruction.

        Rules:
        1. Replace vague references with specifics from the current state.
        2. If the current state already satisfies part of the goal, skip that part.
        3. Keep it as a single short instruction -- do NOT split into sub-steps.
        4. The refined goal must be self-contained.

        Example:
          Current state: "Copied a funny dog meme image, closed the tab,
            Safari is still open."
          Next planned goal: "Open Messenger and send John the funny image"
          Refined: "Open messenger.com in Safari and paste the copied image
            into the chat with John"

        Respond with JSON: {"goal": "<refined>"}
    """)

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Current state: {previous_result}\n\n"
                f"Next planned goal: {original_goal}"
            ),
        },
    ]


def build_validate_goal_messages(
    task: str, goal: str, result: str,
) -> list[dict]:
    """Build messages for goal validation."""
    system_content = textwrap.dedent("""\
        You are a quality-assurance evaluator for a macOS desktop automation agent.
        You will receive the overall task, the specific goal that was just attempted,
        and a summary of what happened. Determine whether the goal was completed
        successfully.

        Look for signs of failure such as:
        - Wrong content was acted on
        - Target app or page was not reached
        - An error or unexpected state is described
        - Goal's core objective was not achieved

        Be lenient on minor deviations -- only flag clear failures.

        Respond with JSON: {"success": true/false, "reason": "<brief explanation>"}
    """)

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Overall task: {task}\n\n"
                f"Goal attempted: {goal}\n\n"
                f"Result: {result}"
            ),
        },
    ]


def build_replan_messages(
    task: str,
    goals_log: list[str],
    current_state: str | None,
) -> list[dict]:
    """Build messages for replanning after multiple goal failures."""
    system_content = textwrap.dedent("""\
        You are a planning assistant for a macOS desktop automation agent.
        The current plan has had too many failures. Given the original task,
        a log of what was tried (with success/failure), and the current screen
        state, produce a NEW ordered list of goals that avoids the failed approaches.

        ## Rules

        1. Do NOT repeat approaches that already failed.
        2. Each goal must be concrete and specific: name the app/window/URL explicitly,
           include the expected visible outcome at the end of each goal.
        3. Goals run sequentially -- later goals can assume earlier ones succeeded.
        4. Use 1-7 goals.
        5. The final goal should produce the end result the user asked for.

        Respond with JSON: {"goals": ["goal 1", ...]}
    """)

    log_text = "\n".join(f"- {entry}" for entry in goals_log) if goals_log else "(none)"
    state_text = current_state or "(unknown)"

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Original task: {task}\n\n"
                f"What was tried:\n{log_text}\n\n"
                f"Current state: {state_text}"
            ),
        },
    ]


def build_goal_self_summary_messages(
    goal: str, recent_results: list[str],
) -> list[dict]:
    """Build messages for goal self-summary."""
    results_text = (
        "\n".join(f"- {r}" for r in recent_results)
        if recent_results
        else "(no actions taken)"
    )

    system_content = textwrap.dedent("""\
        You are a summarization assistant for a macOS desktop automation agent.
        Given a goal and the sequence of action results from executing it,
        produce a brief summary (1-3 sentences) of what was accomplished
        and the current state of the screen / system.
        Focus on the outcome and what is visible or ready for the next step.

        Respond with JSON: {"result": "<summary>"}
    """)

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nAction results:\n{results_text}",
        },
    ]
