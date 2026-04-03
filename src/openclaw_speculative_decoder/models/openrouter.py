"""OpenRouter provider — uses OpenAI-compatible API format."""

from __future__ import annotations

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderFactory
from .openai import OpenAIProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter uses the OpenAI API format; we reuse OpenAIProvider with a custom base URL."""

    provider_name = "openrouter"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._inner = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self._inner.complete(request)
        # Normalize provider name
        response.provider = self.provider_name
        return response


ProviderFactory.register("openrouter", OpenRouterProvider)
