"""OpenAI LLM client with vision and JSON structured output.

Uses the openai SDK directly -- no LangChain dependency.
GPT-4o-mini is natively multimodal and supports json_object response format.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

import openai
from pydantic import BaseModel

from config import MarkConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Thin wrapper around the OpenAI async client."""

    def __init__(self, config: MarkConfig) -> None:
        self.config = config
        self.client = openai.AsyncOpenAI()
        self.model = config.model
        self.temperature = config.temperature
        logger.info("LLM: %s (temp=%.2f)", self.model, self.temperature)

    async def decide(
        self,
        messages: list[dict],
        response_model: type[T],
        image_b64: str | None = None,
    ) -> T:
        """Send messages to the LLM and parse the JSON response into *response_model*.

        If *image_b64* is provided, it is appended to the last user message
        as a base64-encoded image (vision input).
        """
        if image_b64:
            messages = self._attach_image(messages, image_b64)

        t0 = time.monotonic()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        content = response.choices[0].message.content or "{}"
        logger.info("LLM responded in %.0fms (%d chars)", elapsed_ms, len(content))
        logger.debug("LLM raw: %.500s", content)

        data = json.loads(content)
        return response_model.model_validate(data)

    @staticmethod
    def _attach_image(messages: list[dict], image_b64: str) -> list[dict]:
        """Replace the last user message content with a multimodal [text, image] list."""
        messages = [m.copy() for m in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                text = messages[i]["content"]
                if isinstance(text, str):
                    messages[i]["content"] = [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "auto",
                            },
                        },
                    ]
                break
        return messages
