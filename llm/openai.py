"""OpenAI / ChatGPT provider."""

from __future__ import annotations

import logging

import openai

from config import MarkConfig
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Uses the openai SDK with JSON-object response format."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, config: MarkConfig) -> None:
        super().__init__(config)
        self.client = openai.AsyncOpenAI()
        logger.debug("LLM provider: OpenAI (%s, temp=%.2f)", self.model, self.temperature)

    async def _raw_call(
        self,
        messages: list[dict],
        image_b64: str | None,
    ) -> str:
        if image_b64:
            messages = _attach_image(messages, image_b64)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            self._empty_reason = f"finish_reason={response.choices[0].finish_reason}"
            logger.warning("OpenAI returned empty content (%s)", self._empty_reason)
            return ""
        return content


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
