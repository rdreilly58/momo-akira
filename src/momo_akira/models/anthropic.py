"""Anthropic provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

import anthropic

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderFactory


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = self._timer()
        try:
            # Separate system message from messages list (Anthropic API format)
            system = request.system or ""
            messages = request.messages

            # If no explicit system but messages have a system role, extract it
            if not system:
                filtered = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system = str(msg.get("content", ""))
                    else:
                        filtered.append(msg)
                messages = filtered

            kwargs: dict[str, Any] = {
                "model": request.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
            }
            if system:
                kwargs["system"] = system
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            kwargs.update(request.extra_params)

            response = await self._client.messages.create(**kwargs)

            content = ""
            if response.content:
                content = response.content[0].text if hasattr(response.content[0], "text") else ""

            latency_ms = self._timer() - start
            return LLMResponse(
                content=content,
                model_id=request.model_id,
                provider=self.provider_name,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                logprobs=None,  # Anthropic does not expose logprobs
                raw=response.model_dump() if hasattr(response, "model_dump") else {},
                success=True,
            )

        except anthropic.APIError as e:
            latency_ms = self._timer() - start
            return LLMResponse(
                content="",
                model_id=request.model_id,
                provider=self.provider_name,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
        except asyncio.TimeoutError:
            latency_ms = self._timer() - start
            return LLMResponse(
                content="",
                model_id=request.model_id,
                provider=self.provider_name,
                latency_ms=latency_ms,
                success=False,
                error="Request timed out",
            )


ProviderFactory.register("anthropic", AnthropicProvider)
