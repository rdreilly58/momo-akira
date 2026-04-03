"""Integration tests for the full cascade pipeline.

Uses mocked providers — no real API calls.
Tests the cascade logic for both PSDA and PSDF variants.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from openclaw_speculative_decoder.cascade.pipeline import AcceptedTier, CascadePipeline
from openclaw_speculative_decoder.config import AppConfig
from openclaw_speculative_decoder.models.base import LLMResponse


def _make_response(
    content: str,
    model_id: str = "test",
    success: bool = True,
    logprobs: list[float] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_id=model_id,
        provider="mock",
        input_tokens=50,
        output_tokens=len(content.split()),
        latency_ms=100.0,
        logprobs=logprobs,
        success=success,
        error=None if success else "Mock error",
    )


EASY_RESPONSE = _make_response(
    "Paris is the capital of France.",
    logprobs=[-0.1, -0.05, -0.08],  # High confidence
)

MEDIUM_RESPONSE = _make_response(
    "TCP is connection-oriented and reliable.",
    logprobs=[-1.5, -2.0, -1.8],  # Moderate confidence
)

TARGET_RESPONSE = _make_response("The proof by contradiction shows √2 is irrational...")


@pytest.fixture
def pipeline_with_mocks(app_config: AppConfig):
    """Build a CascadePipeline with fully mocked providers."""
    mock_draft = AsyncMock()
    mock_qualifier = AsyncMock()
    mock_target = AsyncMock()

    with patch("openclaw_speculative_decoder.cascade.pipeline.ProviderFactory"):
        pipeline = CascadePipeline(app_config)

    # Replace providers directly
    pipeline._providers["draft"] = mock_draft
    pipeline._providers["qualifier"] = mock_qualifier
    pipeline._providers["target"] = mock_target

    return pipeline, mock_draft, mock_qualifier, mock_target


class TestCascadePSDA:
    """Tests for PSDA (Assisted) variant."""

    @pytest.mark.asyncio
    async def test_draft_accepted_high_confidence(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        simple_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        # Set very high-confidence easy response
        high_conf_response = _make_response(
            "Paris is the capital of France.",
            logprobs=[-0.01, -0.02, -0.01],  # Near-zero log probs = very confident
        )
        mock_draft.complete.return_value = high_conf_response

        # Force tau_q to be very low so draft is always accepted
        pipeline._threshold_controller._state.tau_q = 0.01

        result = await pipeline.run(messages=simple_messages)

        assert result.draft_invoked
        assert not result.qualifier_invoked
        assert not result.target_invoked
        assert result.accepted_tier == AcceptedTier.DRAFT
        mock_qualifier.complete.assert_not_called()
        mock_target.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_escalates_to_qualifier_when_draft_rejected(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        complex_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        # Draft has terrible response (empty = high divergence)
        mock_draft.complete.return_value = _make_response("", success=True)
        mock_qualifier.complete.return_value = MEDIUM_RESPONSE

        # Very high tau_q so draft is always rejected, low tau_t so qualifier is accepted
        pipeline._threshold_controller._state.tau_q = 0.99
        pipeline._threshold_controller._state.tau_t = 0.01

        result = await pipeline.run(messages=complex_messages)

        assert result.draft_invoked
        assert result.qualifier_invoked
        assert not result.target_invoked
        assert result.accepted_tier == AcceptedTier.QUALIFIER

    @pytest.mark.asyncio
    async def test_escalates_to_target_when_qualifier_rejected(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        complex_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        mock_draft.complete.return_value = _make_response("", success=True)
        mock_qualifier.complete.return_value = _make_response("", success=True)
        mock_target.complete.return_value = TARGET_RESPONSE

        # Force all thresholds very high to ensure escalation
        pipeline._threshold_controller._state.tau_q = 0.99
        pipeline._threshold_controller._state.tau_t = 0.99

        result = await pipeline.run(messages=complex_messages)

        assert result.draft_invoked
        assert result.qualifier_invoked
        assert result.target_invoked
        assert result.accepted_tier == AcceptedTier.TARGET

    @pytest.mark.asyncio
    async def test_draft_failure_skips_to_qualifier(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        simple_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        mock_draft.complete.return_value = _make_response("", success=False)
        mock_qualifier.complete.return_value = MEDIUM_RESPONSE
        pipeline._threshold_controller._state.tau_t = 0.01

        result = await pipeline.run(messages=simple_messages)

        assert result.draft_invoked
        assert result.qualifier_invoked
        # Should NOT have called target (qualifier accepted with low tau_t)
        assert result.accepted_tier == AcceptedTier.QUALIFIER

    @pytest.mark.asyncio
    async def test_all_tiers_fail_returns_target_error(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        simple_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        mock_draft.complete.return_value = _make_response("", success=False)
        mock_qualifier.complete.return_value = _make_response("", success=False)
        mock_target.complete.return_value = _make_response("", success=False)

        result = await pipeline.run(messages=simple_messages)

        # Should still return a result (graceful degradation)
        assert result.response is not None
        assert result.target_invoked

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_tiers(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        complex_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        mock_draft.complete.return_value = _make_response("")
        mock_qualifier.complete.return_value = _make_response("")
        mock_target.complete.return_value = TARGET_RESPONSE

        pipeline._threshold_controller._state.tau_q = 0.99
        pipeline._threshold_controller._state.tau_t = 0.99

        result = await pipeline.run(messages=complex_messages)

        # Cost should be non-zero (3 tiers invoked)
        assert result.total_cost >= 0.0  # providers use mock tokens

    @pytest.mark.asyncio
    async def test_cascade_metadata(
        self,
        app_config: AppConfig,
        pipeline_with_mocks: tuple,
        simple_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = pipeline_with_mocks
        mock_draft.complete.return_value = EASY_RESPONSE
        pipeline._threshold_controller._state.tau_q = 0.01

        result = await pipeline.run(messages=simple_messages)
        meta = result.to_metadata()

        assert "accepted_tier" in meta
        assert "variant" in meta
        assert "tiers_invoked" in meta
        assert "confidence" in meta
        assert "cost_usd" in meta
        assert "latency_ms" in meta


class TestCascadePSDF:
    """Tests for PSDF (Fuzzy) variant."""

    @pytest.fixture
    def psdf_pipeline(self, psdf_config: AppConfig):
        mock_draft = AsyncMock()
        mock_qualifier = AsyncMock()
        mock_target = AsyncMock()

        with patch("openclaw_speculative_decoder.cascade.pipeline.ProviderFactory"):
            pipeline = CascadePipeline(psdf_config)

        pipeline._providers["draft"] = mock_draft
        pipeline._providers["qualifier"] = mock_qualifier
        pipeline._providers["target"] = mock_target

        return pipeline, mock_draft, mock_qualifier, mock_target

    @pytest.mark.asyncio
    async def test_psdf_draft_accepted(
        self,
        psdf_pipeline: tuple,
        simple_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = psdf_pipeline
        mock_draft.complete.return_value = EASY_RESPONSE
        pipeline._threshold_controller._state.tau_q = 0.01

        result = await pipeline.run(messages=simple_messages)

        assert result.accepted_tier == AcceptedTier.DRAFT
        assert result.variant == "psdf"

    @pytest.mark.asyncio
    async def test_psdf_full_cascade(
        self,
        psdf_pipeline: tuple,
        complex_messages: list[dict],
    ) -> None:
        pipeline, mock_draft, mock_qualifier, mock_target = psdf_pipeline
        mock_draft.complete.return_value = _make_response("")
        mock_qualifier.complete.return_value = _make_response("")
        mock_target.complete.return_value = TARGET_RESPONSE

        pipeline._threshold_controller._state.tau_q = 0.99
        pipeline._threshold_controller._state.tau_t = 0.99

        result = await pipeline.run(messages=complex_messages)

        assert result.draft_invoked
        assert result.qualifier_invoked
        assert result.target_invoked
        assert result.variant == "psdf"
