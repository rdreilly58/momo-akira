"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw_speculative_decoder.cascade.pipeline import CascadePipeline
from openclaw_speculative_decoder.config import (
    AdaptiveThresholdConfig,
    AppConfig,
    CascadeConfig,
    ConfidenceConfig,
    LoggingConfig,
    ModelConfig,
    ModelsConfig,
    ProvidersConfig,
    ServerConfig,
)
from openclaw_speculative_decoder.models.base import LLMResponse


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def draft_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="claude-haiku-test",
        provider="anthropic",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        expected_latency_ms=500,
        timeout_s=10,
    )


@pytest.fixture
def qualifier_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="claude-sonnet-test",
        provider="anthropic",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        expected_latency_ms=1500,
        timeout_s=20,
    )


@pytest.fixture
def target_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="claude-opus-test",
        provider="anthropic",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        expected_latency_ms=4000,
        timeout_s=30,
    )


@pytest.fixture
def app_config(
    draft_model_config: ModelConfig,
    qualifier_model_config: ModelConfig,
    target_model_config: ModelConfig,
) -> AppConfig:
    return AppConfig(
        server=ServerConfig(),
        models=ModelsConfig(
            draft=draft_model_config,
            qualifier=qualifier_model_config,
            target=target_model_config,
        ),
        cascade=CascadeConfig(variant="psda", tau_q=0.35, tau_t=0.50),
        confidence=ConfidenceConfig(
            prompt_complexity_weight=0.30,
            response_coherence_weight=0.40,
            logprob_weight=0.30,
            use_logprobs=False,
            use_self_scoring=False,
        ),
        adaptive_threshold=AdaptiveThresholdConfig(
            enabled=False,  # Disabled for tests to keep thresholds stable
            warmup_requests=1000,
        ),
        logging=LoggingConfig(enabled=False),
        providers=ProvidersConfig(),
    )


@pytest.fixture
def psdf_config(app_config: AppConfig) -> AppConfig:
    """Config with PSDF variant."""
    app_config.cascade.variant = "psdf"
    return app_config


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


def _make_response(content: str, model_id: str = "test", success: bool = True) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_id=model_id,
        provider="mock",
        input_tokens=50,
        output_tokens=len(content.split()),
        latency_ms=100.0,
        logprobs=None,
        success=success,
        error=None if success else "Mock error",
    )


EASY_RESPONSE = _make_response("Paris is the capital of France.")
MEDIUM_RESPONSE = _make_response(
    "TCP is connection-oriented and reliable. UDP is connectionless and fast."
)
HARD_RESPONSE = _make_response(
    "The proof proceeds by contradiction. Assume √2 = p/q in lowest terms..."
)
FAILED_RESPONSE = _make_response("", success=False)


# ---------------------------------------------------------------------------
# Pipeline fixture with mocked providers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline(app_config: AppConfig):
    """CascadePipeline with all providers mocked out."""
    with patch("openclaw_speculative_decoder.cascade.pipeline.ProviderFactory") as mock_factory:
        mock_draft_provider = AsyncMock()
        mock_qualifier_provider = AsyncMock()
        mock_target_provider = AsyncMock()

        def create_side_effect(provider_name, api_key, base_url=None):
            # Return different mocks based on call order
            call_count = mock_factory.create.call_count
            return {
                1: mock_draft_provider,
                2: mock_qualifier_provider,
                3: mock_target_provider,
            }.get(call_count, mock_target_provider)

        mock_factory.create.side_effect = create_side_effect

        pipeline = CascadePipeline(app_config)
        # Directly assign providers
        pipeline._providers["draft"] = mock_draft_provider
        pipeline._providers["qualifier"] = mock_qualifier_provider
        pipeline._providers["target"] = mock_target_provider

        yield pipeline, mock_draft_provider, mock_qualifier_provider, mock_target_provider


# ---------------------------------------------------------------------------
# Simple messages fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "What is the capital of France?"}]


@pytest.fixture
def complex_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                "Implement a balanced binary search tree in Python. "
                "Include insert, delete, search, and in-order traversal. "
                "Analyze the time complexity of each operation step by step."
            ),
        }
    ]
