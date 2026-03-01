"""Google Gemini provider."""

from __future__ import annotations

import base64
import logging

from google import genai
from google.genai import types

from config import MarkConfig
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Uses the google-genai SDK with JSON response mime-type."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, config: MarkConfig) -> None:
        super().__init__(config)
        self.client = genai.Client()
        logger.debug("LLM provider: Gemini (%s, temp=%.2f)", self.model, self.temperature)

    async def _raw_call(
        self,
        messages: list[dict],
        image_b64: str | None,
    ) -> str:
        system_instruction, contents = _convert_messages(messages, image_b64)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

        try:
            text = response.text
        except (ValueError, AttributeError) as exc:
            logger.debug("response.text raised %s: %s", type(exc).__name__, exc)
            text = None

        if not text or not text.strip():
            text = _extract_text_from_parts(response)

        if not text or not text.strip():
            finish = getattr(response.candidates[0], "finish_reason", "unknown") if response.candidates else "no_candidates"
            safety = getattr(response.candidates[0], "safety_ratings", None) if response.candidates else None
            self._empty_reason = f"finish_reason={finish}, safety={safety}"
            logger.warning("Gemini returned empty content (%s)", self._empty_reason)
            return ""
        return text


def _extract_text_from_parts(response) -> str:
    """Extract text directly from response parts, bypassing the .text property.

    Handles thinking-model responses (gemini-2.5-*) where .text may return
    empty even though the model produced output in non-thought parts.
    """
    if not response.candidates:
        return ""
    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []

    thought_count = 0
    texts: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            thought_count += 1
            continue
        part_text = getattr(part, "text", None)
        if part_text:
            texts.append(part_text)

    if thought_count and not texts:
        logger.warning(
            "Gemini returned %d thought part(s) but no output text -- "
            "thinking model may need an explicit output prompt",
            thought_count,
        )
    elif not parts:
        logger.debug("Gemini response had candidates but no parts")

    return "".join(texts)


def _convert_messages(
    messages: list[dict],
    image_b64: str | None,
) -> tuple[str | None, list[types.Content]]:
    """Translate OpenAI-style messages to Gemini contents.

    Returns (system_instruction, contents).
    System messages are extracted; "assistant" role becomes "model".
    The image (if any) is attached to the last user turn.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []
    last_user_idx: int | None = None

    for msg in messages:
        role = msg["role"]
        text = msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(
                part.get("text", "") for part in text if part.get("type") == "text"
            )

        if role == "system":
            system_parts.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append(types.Content(
            role=gemini_role,
            parts=[types.Part.from_text(text=text)],
        ))
        if gemini_role == "user":
            last_user_idx = len(contents) - 1

    if image_b64 and last_user_idx is not None:
        image_bytes = base64.b64decode(image_b64)
        contents[last_user_idx].parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents
