"""LLM provider factory.

Usage::

    from llm import create_llm_client
    client = create_llm_client(config)       # returns the right provider
    result = await client.decide(messages, ResponseModel, image_b64=img)
"""

from __future__ import annotations

from config import MarkConfig
from llm.base import LLMProvider

_PROVIDERS = ("openai", "gemini", "ollama")


def create_llm_client(config: MarkConfig) -> LLMProvider:
    """Instantiate the LLM provider specified by *config.provider*."""
    match config.provider:
        case "openai":
            from llm.openai import OpenAIProvider
            return OpenAIProvider(config)
        case "gemini":
            from llm.gemini import GeminiProvider
            return GeminiProvider(config)
        case "ollama":
            from llm.ollama import OllamaProvider
            return OllamaProvider(config)
        case _:
            raise ValueError(
                f"Unknown LLM provider {config.provider!r}. "
                f"Choose from: {', '.join(_PROVIDERS)}"
            )
