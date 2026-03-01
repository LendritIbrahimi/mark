"""Local Ollama provider."""

from __future__ import annotations

import logging

import ollama

from config import MarkConfig
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Uses the ollama SDK talking to a locally-running Ollama server."""

    DEFAULT_MODEL = "llama3.2-vision"

    def __init__(self, config: MarkConfig) -> None:
        super().__init__(config)
        self.client = ollama.AsyncClient()
        logger.debug("LLM provider: Ollama (%s, temp=%.2f)", self.model, self.temperature)

    async def _raw_call(
        self,
        messages: list[dict],
        image_b64: str | None,
    ) -> str:
        msgs = _convert_messages(messages, image_b64)

        response = await self.client.chat(
            model=self.model,
            messages=msgs,
            format="json",
            options={"temperature": self.temperature},
        )
        content = response.message.content
        if not content or not content.strip():
            self._empty_reason = f"done_reason={getattr(response, 'done_reason', 'unknown')}"
            logger.warning("Ollama returned empty content (%s)", self._empty_reason)
            return ""
        return content


def _convert_messages(
    messages: list[dict],
    image_b64: str | None,
) -> list[dict]:
    """Build Ollama message list, attaching the image to the last user turn.

    Ollama accepts the same role names as OpenAI (system / user / assistant).
    Images are passed as a ``images`` list of raw base64 strings on the message.
    """
    msgs: list[dict] = []
    last_user_idx: int | None = None

    for msg in messages:
        text = msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(
                part.get("text", "") for part in text if part.get("type") == "text"
            )

        entry: dict = {"role": msg["role"], "content": text}
        msgs.append(entry)

        if msg["role"] == "user":
            last_user_idx = len(msgs) - 1

    if image_b64 and last_user_idx is not None:
        msgs[last_user_idx]["images"] = [image_b64]

    return msgs
