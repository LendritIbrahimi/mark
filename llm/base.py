"""Abstract base class for LLM providers.

Every provider implements `_raw_call` to hit its specific API.
The shared `decide` method handles timing, JSON parsing, and
Pydantic validation so each provider only worries about the
wire format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from config import MarkConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Provider-agnostic interface used by the agent loop.

    Subclasses must set ``DEFAULT_MODEL`` and implement ``_raw_call``.
    """

    DEFAULT_MODEL: str

    def __init__(self, config: MarkConfig) -> None:
        self.model = config.model or self.DEFAULT_MODEL
        self.temperature = config.temperature
        self.last_raw_response: str = ""

    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.5
    CALL_TIMEOUT = 30.0

    async def decide(
        self,
        messages: list[dict],
        response_model: type[T],
        image_b64: str | None = None,
    ) -> T:
        """Send *messages* (with optional screenshot) and return a parsed Pydantic model.

        Retries up to MAX_RETRIES times on empty, malformed, or timed-out responses.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._empty_reason = None
            t0 = time.monotonic()
            try:
                content = await asyncio.wait_for(
                    self._raw_call(messages, image_b64),
                    timeout=self.CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - t0) * 1000
                last_error = TimeoutError(f"LLM call timed out after {self.CALL_TIMEOUT:.0f}s")
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF * attempt
                    logger.warning("LLM call timed out at %.0fms (attempt %d/%d), retrying in %.1fs...", elapsed_ms, attempt, self.MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                raise last_error
            self.last_raw_response = content
            elapsed_ms = (time.monotonic() - t0) * 1000

            logger.debug("LLM responded in %.0fms (%d chars)", elapsed_ms, len(content))
            logger.debug("LLM raw: %s", content)

            if not content or not content.strip():
                reason = self._empty_reason or "unknown"
                last_error = ValueError(f"LLM returned empty response ({reason})")
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF * attempt
                    logger.warning("Empty LLM response (attempt %d/%d), retrying in %.1fs...", attempt, self.MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = ValueError(f"LLM returned invalid JSON: {exc}")
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF * attempt
                    logger.warning("Invalid JSON from LLM (attempt %d/%d), retrying in %.1fs...", attempt, self.MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                logger.error("LLM returned invalid JSON: %s", content)
                raise last_error from exc

            return response_model.model_validate(data)

        raise last_error or RuntimeError("LLM retries exhausted")

    @abstractmethod
    async def _raw_call(
        self,
        messages: list[dict],
        image_b64: str | None,
    ) -> str:
        """Hit the provider API and return the raw JSON string."""
