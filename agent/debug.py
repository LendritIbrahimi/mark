"""Debug logging utilities for per-step traces and screenshots."""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from typing import Any


def _sanitize(text: str, max_len: int = 50) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return cleaned[:max_len]


def create_session_dir(task: str, base_dir: str = "debug_logs") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}__{_sanitize(task)}"
    session_dir = os.path.join(base_dir, folder_name)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def create_step_dir(session_dir: str, step: int, goal_idx: int = 1) -> str:
    goal_dir = os.path.join(session_dir, f"goal_{goal_idx}")
    step_dir = os.path.join(goal_dir, f"step_{step:03d}")
    os.makedirs(step_dir, exist_ok=True)
    return step_dir


def write_step_trace(step_dir: str, data: dict[str, Any]) -> None:
    path = os.path.join(step_dir, "step_trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_b64_image(b64_data: str, path: str) -> None:
    raw = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(raw)
