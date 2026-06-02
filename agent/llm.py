"""OpenAI LLM client with JSON mode and Pydantic validation."""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from agent.config import DEFAULT_MODEL, MarkConfig, is_reasoning_model

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)


class OpenAILLM:
    """OpenAI chat completion client with SDK-managed retries."""

    def __init__(self, config: MarkConfig) -> None:
        self.model = config.model or DEFAULT_MODEL
        self.temperature = config.temperature
        self.reasoning_effort = config.reasoning_effort
        self.is_reasoning = is_reasoning_model(self.model)
        self.last_raw_response: str = ""
        self.client = AsyncOpenAI(
            max_retries=config.llm_max_retries,
            timeout=config.llm_timeout,
        )

    async def decide(
            self,
            messages: list[dict],
            response_model: type[_T],
            image_b64: str | None = None,
    ) -> _T:
        """Send messages and return a parsed Pydantic model."""
        openai_messages = _prepare_messages(
            messages, image_b64, reasoning_model=self.is_reasoning,
        )

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

        choice = response.choices[0] if response.choices else None
        content = (choice.message.content or "") if choice else ""
        self.last_raw_response = content

        if not content.strip():
            raise ValueError("LLM returned empty response")

        data = json.loads(content)
        return response_model.model_validate(data)


def _prepare_messages(
        messages: list[dict],
        image_b64: str | None,
        *,
        reasoning_model: bool = False,
) -> list[dict]:
    prepared = list(messages)

    if reasoning_model:
        prepared = [
            {**msg, "role": "developer"}
            if msg.get("role") == "system"
            else msg
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
                part.get("text", "")
                for part in text
                if part.get("type") == "text"
            )
        prepared[last_user_idx] = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/jpeg;base64,"
                            f"{image_b64}"
                        ),
                        "detail": "high",
                    },
                },
            ],
        }

    return prepared
