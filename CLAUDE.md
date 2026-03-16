# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**mark** is a macOS desktop automation agent. It uses an LLM (OpenAI) to perceive the screen (via screenshots + macOS Accessibility API) and execute mouse/keyboard actions to accomplish natural-language tasks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires an `OPENAI_API_KEY` in a `.env` file (loaded automatically via `python-dotenv`).

## Running

```bash
# CLI
python -m main "open Safari and search for cats"

# CLI flags
python -m main "task" --model gpt-4o --no-orchestrator --no-vision --reasoning-effort high

# GUI (customtkinter)
python -m ui.app

# Build macOS .app bundle (no Terminal window)
bash create_app.sh
# Then double-click mark.app
```

There is no test suite.

## Architecture

The system has three layers that communicate in a pipeline:

```
main.py / ui/app.py
    └─► agent/orchestrator.py  (task decomposition)
            └─► agent/loop.py  (perceive-think-act cycle)
                    ├─► servers/vision/server.py  (MCP subprocess)
                    └─► servers/action/server.py  (MCP subprocess)
```

### MCP Servers (subprocesses)

Both servers are launched as subprocesses via stdio MCP. Each run is a separate Python process.

- **`servers/vision/server.py`** — single tool `observe()`: captures a screenshot, queries the macOS Accessibility API for UI elements, draws numbered bounding boxes on the image, and returns `{image, elements, element_positions, scale, backing_scale}` as JSON.
- **`servers/action/server.py`** — exposes mouse and keyboard tools: `click`, `double_click`, `right_click`, `hover_at`, `drag_to`, `scroll_at`, `type_text`, `press_key`, `hotkey_press`.

Server stderr is captured to `.mcp_vision.log` / `.mcp_action.log` and logged on errors.

### Agent Core (`agent/`)

- **`orchestrator.py`** — `Orchestrator`: triages a task as simple/complex, optionally decomposes it into ordered sub-goals (using a planner LLM), then executes each goal via `AgentLoop`, validates the result, and retries on failure. The planner uses `MarkConfig.orchestrator_model` if set, falling back to the main model.

- **`loop.py`** — `AgentLoop`: the perceive → think → act loop. Each step:
  1. Calls `observe()` on the vision MCP to get a screenshot + element list
  2. Builds a prompt from `StateManager` state and sends it to the LLM
  3. Parses `StepResponse` (observation, thought, list of `ActionCall`s)
  4. Dispatches each action through `ActionExecutor`
  - Stops on: `done()` action, too many consecutive failures, stale/looping detection.
  - History is trimmed to `max_messages` and older user messages are compressed (element lists replaced with counts).

- **`executor.py`** — `ActionExecutor`: maps agent action names (e.g. `click`) to MCP tool names (e.g. `click`), resolves `element_id` parameters to `(x, y)` coordinates by looking up `StateManager.element_positions`. `wait` and `done` are handled locally without MCP calls.

- **`llm.py`** — `OpenAILLM`: wraps `AsyncOpenAI` with retry logic, always uses `response_format: json_object`, returns a validated Pydantic model. For reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`), the `system` role becomes `developer` and `reasoning_effort` is passed instead of `temperature`.

- **`state.py`** — `StateManager`: mutable per-run state. Tracks element positions (id → x/y), recent/all action results, loop detection (repeated actions, unchanged element counts), and `user_guidance` injected mid-run from the UI.

- **`callbacks.py`** — `AgentCallbacks`: optional observer hooks (`on_step_start`, `on_think`, `on_action_result`, `on_goal_start`, `on_goal_end`, `on_decompose`, `on_done`). Also holds `pause_event` (threading.Event) and `stop_requested` flag. Callables are invoked from the asyncio thread; the UI marshals them to its own thread via `root.after()`.

- **`config.py`** — `MarkConfig` dataclass: all tuneable parameters. Default model is `gpt-5-nano`.

- **`prompts.py`** — All prompt templates as functions returning `list[dict]` message arrays.

- **`debug.py`** — Per-session/step debug output saved to `debug_logs/<timestamp>__<task>/step_NNN/` containing `screenshot_labeled.jpg`, `elements.txt`, `llm_response.json`, and `step_trace.json`.

### UI (`ui/`)

`MarkApp` (customtkinter) runs the agent in a background `threading.Thread` with its own `asyncio` event loop. Callbacks use `self.after(0, ...)` to marshal updates back to the Tkinter main thread. The UI supports pause/resume (via `pause_event`), stop, and mid-run guidance injection.

## Key conventions

- All LLM calls return Pydantic models via JSON mode — never free-form text parsing.
- Agent actions always use `element_id` references (never raw coordinates); `ActionExecutor` resolves them to `(x, y)` at dispatch time.
- The vision server uses `NSApplicationActivationPolicyProhibited` to avoid appearing in the Dock.
- `connect_mcp()` in `mcp_client.py` is an async context manager; both MCP connections must be held open for the duration of a run.
