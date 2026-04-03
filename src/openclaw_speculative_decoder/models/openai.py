"""OpenAI provider implementation (also compatible with OpenAI-format APIs)."""

from __future__ import annotations

import asyncio
from typing import Any

import openai as openai_lib

from .base import LLMProvider, LLMRequest, LLMResponse, ProviderFactory


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai_lib.AsyncOpenAI(**kwargs)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = self._timer()
        try:
            messages = list(request.messages)

            # Prepend system message if provided separately
            if request.system:
                messages = [{"role": "system", "content": request.system}] + messages

            kwargs: dict[str, Any] = {
                "model": request.model_id,
                "messages": messages,
            }
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.request_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1
            kwargs.update(request.extra_params)

            response = await self._client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            content = choice.message.content or ""

            # Extract per-token logprobs if available
            logprobs: list[float] | None = None
            if choice.logprobs and choice.logprobs.content:
                logprobs = [t.logprob for t in choice.logprobs.content]

            latency_ms = self._timer() - start
            usage = response.usage
            return LLMResponse(
                content=content,
                model_id=request.model_id,
                provider=self.provider_name,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                logprobs=logprobs,
                raw=response.model_dump() if hasattr(response, "model_dump") else {},
                success=True,
            )

        except openai_lib.APIError as e:
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


ProviderFactory.register("openai", OpenAIProvider)
