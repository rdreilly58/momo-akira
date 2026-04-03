"""Abstract LLM provider interface.

All providers implement the same interface so models are fully swappable
without any code changes — just update config.yaml.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    """Normalized request to any LLM provider."""

    model_id: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    system: str | None = None
    # Extra provider-specific params forwarded as-is
    extra_params: dict[str, Any] = field(default_factory=dict)
    # Whether to request logprobs if the provider supports it
    request_logprobs: bool = False


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    model_id: str
    provider: str
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    # Latency
    latency_ms: float = 0.0
    # Logprobs if available (list of per-token log probabilities)
    logprobs: list[float] | None = None
    # Raw provider response for debugging
    raw: dict[str, Any] = field(default_factory=dict)
    # Whether the request succeeded
    success: bool = True
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, cost_per_1k_input: float, cost_per_1k_output: float) -> float:
        return (
            self.input_tokens / 1000 * cost_per_1k_input
            + self.output_tokens / 1000 * cost_per_1k_output
        )


class LLMProvider(abc.ABC):
    """Abstract base class for all LLM providers."""

    provider_name: str = ""

    @abc.abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return a normalized response."""

    @staticmethod
    def _timer() -> float:
        return time.monotonic() * 1000  # ms


class ProviderFactory:
    """Creates provider instances from config."""

    _registry: dict[str, type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        cls._registry[name] = provider_cls

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str | None,
        base_url: str | None = None,
    ) -> LLMProvider:
        if provider_name not in cls._registry:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider_name](api_key=api_key, base_url=base_url)
