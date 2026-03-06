"""OpenAI LLM client with retry logic, JSON parsing, and Pydantic validation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from config import MarkConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "o4-mini"
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5

_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Return True if the model is an OpenAI reasoning (o-series) model."""
    base = model.split("-")[0] if "-" in model else model
    return base in _REASONING_PREFIXES


class OpenAILLM:
    """OpenAI LLM client used by the agent loop.

    Handles retries, timeouts, JSON parsing, and Pydantic validation.
    """

    def __init__(self, config: MarkConfig) -> None:
        self.model = config.model or DEFAULT_MODEL
        self.temperature = config.temperature
        self.reasoning_effort = config.reasoning_effort
        self.is_reasoning = _is_reasoning_model(self.model)
        self.call_timeout = config.llm_timeout
        self.last_raw_response: str = ""
        self.last_response_metadata: dict | None = None
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.client = AsyncOpenAI()
        if self.is_reasoning:
            logger.debug("LLM: OpenAI (%s, reasoning_effort=%s)", self.model, self.reasoning_effort)
        else:
            logger.debug("LLM: OpenAI (%s, temp=%.2f)", self.model, self.temperature)

    async def decide(
        self,
        messages: list[dict],
        response_model: type[T],
        image_b64: str | None = None,
    ) -> T:
        """Send messages (with optional screenshot) and return a parsed Pydantic model.

        Retries up to MAX_RETRIES on empty, malformed, or timed-out responses.
        """
        last_error: Exception | None = None
        retry_t0 = time.monotonic()

        for attempt in range(1, MAX_RETRIES + 1):
            self.total_calls += 1
            t0 = time.monotonic()
            try:
                content = await asyncio.wait_for(
                    self._call_openai(messages, image_b64),
                    timeout=self.call_timeout,
                )
            except asyncio.TimeoutError:
                elapsed_ms = (time.monotonic() - t0) * 1000
                last_error = TimeoutError(f"LLM call timed out after {self.call_timeout:.0f}s")
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF * attempt
                    logger.warning("LLM call timed out at %.0fms (attempt %d/%d), retrying in %.1fs...", elapsed_ms, attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            self.last_raw_response = content
            elapsed_ms = (time.monotonic() - t0) * 1000

            logger.debug("LLM responded in %.0fms (%d chars)", elapsed_ms, len(content))
            logger.debug("LLM raw: %s", content)

            if not content or not content.strip():
                last_error = ValueError("LLM returned empty response")
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF * attempt
                    logger.warning("Empty response from OpenAI (attempt %d/%d), retrying in %.1fs...", attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = ValueError(f"LLM returned invalid JSON: {exc}")
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF * attempt
                    logger.warning("Invalid JSON from LLM (attempt %d/%d), retrying in %.1fs...", attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                logger.error("LLM returned invalid JSON: %s", content)
                raise last_error from exc

            self.successful_calls += 1
            if attempt > 1:
                total_ms = (time.monotonic() - retry_t0) * 1000
                logger.info("LLM recovered on attempt %d/%d (%.0fms total)", attempt, MAX_RETRIES, total_ms)
            return response_model.model_validate(data)

        total_ms = (time.monotonic() - retry_t0) * 1000
        logger.error("All %d LLM retries exhausted in %.0fms", MAX_RETRIES, total_ms)
        raise last_error or RuntimeError("LLM retries exhausted")

    # -- OpenAI API --

    async def _call_openai(
        self,
        messages: list[dict],
        image_b64: str | None,
    ) -> str:
        openai_messages = _prepare_messages(messages, image_b64, reasoning_model=self.is_reasoning)

        kwargs: dict = {
            "model": self.model,
            "messages": openai_messages,
            "response_format": {"type": "json_object"},
        }
        if self.is_reasoning:
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature

        response = await self.client.chat.completions.create(**kwargs)

        self.last_response_metadata = _dump_response_metadata(response)

        choice = response.choices[0] if response.choices else None
        if choice is None:
            logger.warning("OpenAI returned no choices")
            logger.warning("Full response metadata: %s", json.dumps(self.last_response_metadata, indent=2, default=str))
            return ""

        text = choice.message.content or ""

        if not text.strip():
            finish = choice.finish_reason
            logger.warning("OpenAI returned empty content (finish_reason=%s)", finish)
            logger.warning("Full response metadata: %s", json.dumps(self.last_response_metadata, indent=2, default=str))

        return text


# -- Helpers --


def _prepare_messages(
    messages: list[dict],
    image_b64: str | None,
    *,
    reasoning_model: bool = False,
) -> list[dict]:
    """Convert internal messages to OpenAI format.

    Transformations applied:
    - Reasoning models: ``system`` role becomes ``developer``.
    - When *image_b64* is provided, the last user turn's content is converted
      to a content array with an attached image.
    """
    prepared = list(messages)

    if reasoning_model:
        prepared = [
            {**msg, "role": "developer"} if msg.get("role") == "system" else msg
            for msg in prepared
        ]

    last_user_idx: int | None = None
    for i, msg in enumerate(prepared):
        if msg.get("role") == "user":
            last_user_idx = i

    if image_b64 and last_user_idx is not None:
        user_msg = prepared[last_user_idx]
        text = user_msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(
                part.get("text", "") for part in text if part.get("type") == "text"
            )
        prepared[last_user_idx] = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    },
                },
            ],
        }

    return prepared


def _dump_response_metadata(response) -> dict:
    """Extract useful metadata from an OpenAI ChatCompletion response."""
    meta: dict = {}

    meta["model"] = getattr(response, "model", None)
    meta["system_fingerprint"] = getattr(response, "system_fingerprint", None)

    usage = getattr(response, "usage", None)
    if usage:
        meta["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            meta["usage"]["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

    choices = getattr(response, "choices", None) or []
    meta["choices"] = []
    for choice in choices:
        meta["choices"].append({
            "index": getattr(choice, "index", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "content_length": len(choice.message.content) if choice.message and choice.message.content else 0,
        })

    return meta
