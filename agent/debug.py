"""Debug logging utilities -- per-step traces, screenshots, run log."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = "debug_logs"


def _sanitize(text: str, max_len: int = 50) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return cleaned[:max_len]


def create_session_dir(task: str, base_dir: str = DEFAULT_BASE_DIR) -> str:
    """Create and return the session directory for this run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}__{_sanitize(task)}"
    session_dir = os.path.join(base_dir, folder_name)
    os.makedirs(session_dir, exist_ok=True)
    logger.info("Debug session: %s", session_dir)
    return session_dir


def create_step_dir(session_dir: str, step: int) -> str:
    """Create and return step_NNN/ under the session directory."""
    step_dir = os.path.join(session_dir, f"step_{step:03d}")
    os.makedirs(step_dir, exist_ok=True)
    return step_dir


def setup_file_logger(session_dir: str) -> logging.Handler:
    """Add a FileHandler that writes to run.log in the session directory."""
    log_path = os.path.join(session_dir, "run.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def remove_file_logger(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)


def write_step_trace(step_dir: str, data: dict[str, Any]) -> None:
    """Write step_trace.json into the step directory."""
    path = os.path.join(step_dir, "step_trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_b64_image(b64_data: str, path: str) -> None:
    """Decode base64 and save as a file."""
    import base64
    raw = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(raw)
