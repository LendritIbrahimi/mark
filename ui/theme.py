"""Visual theme, model catalog, and small UI helpers."""

from __future__ import annotations

import customtkinter as ctk

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

WIDTH = 500
PAD = 4
MONO = "Menlo"

MODELS = [
    "gpt-5.4", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "o4-mini", "o3",
    "gpt-4o", "gpt-4o-mini",
    "computer-use-preview",
]
REASONING_LEVELS = ["low", "medium", "high"]

MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "computer-use-preview": (3.00, 12.00),
}

# ---------------------------------------------------------------------------
# Monochrome palette
# ---------------------------------------------------------------------------

BG = "#0c0c0c"
SURFACE = "#161616"
BORDER = "#2a2a2a"
HIGHLIGHT = "#1e1e1e"

FG = "#d4d4d4"
FG_DIM = "#777777"
FG_FAINT = "#444444"
FG_GREEN = "#4ade80"
FG_RED = "#f87171"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def trunc(text: str, limit: int = 100) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def unwrap_exc(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [unwrap_exc(e) for e in exc.exceptions]
        return "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"
