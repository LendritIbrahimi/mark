"""Color-coded console formatter for the agent loop.

Provides a clean, scannable terminal output:
  - Step headers  [Step N]  in bold cyan
  - Actions       ->        in green
  - Errors                  in red
  - Everything else         in default color

Respects NO_COLOR env var and non-TTY streams.
"""

from __future__ import annotations

import logging
import os
import sys


class _Ansi:
    """ANSI escape helpers -- disabled when colors are unsupported."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold_cyan(self, text: str) -> str:
        return self._wrap("1;36", text)

    def green(self, text: str) -> str:
        return self._wrap("0;32", text)

    def red(self, text: str) -> str:
        return self._wrap("0;31", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)


def _colors_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stderr, "isatty"):
        return False
    return sys.stderr.isatty()


class ConsoleFormatter(logging.Formatter):
    """Compact, color-coded formatter for non-verbose console output.

    Pattern matching on the message content determines the style:
      [Step ...]   -> bold cyan with timestamp
      -> / →       -> green, no timestamp
      ERROR/WARN   -> red, with timestamp
      others       -> default, no timestamp
    """

    def __init__(self) -> None:
        super().__init__()
        self.ansi = _Ansi(_colors_supported())

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        if record.levelno >= logging.ERROR:
            return self.ansi.red(f"ERROR: {msg}")

        if record.levelno >= logging.WARNING:
            return self.ansi.red(f"WARN: {msg}")

        if msg.startswith("[Step "):
            return self.ansi.bold_cyan(msg)

        if msg.lstrip().startswith(("\u2192", "->")):
            return self.ansi.green(f"  {msg.strip()}")

        return msg
