"""Tests for confidence scoring."""

from __future__ import annotations

import pytest

from openclaw_speculative_decoder.cascade.confidence import ConfidenceScore, score_response
from openclaw_speculative_decoder.config import ConfidenceConfig


@pytest.fixture
def confidence_config() -> ConfidenceConfig:
    return ConfidenceConfig(
        prompt_complexity_weight=0.30,
        response_coherence_weight=0.40,
        logprob_weight=0.30,
        use_logprobs=True,
        use_self_scoring=False,
    )


class TestConfidenceScore:
    def test_accepts_at_threshold(self) -> None:
        cs = ConfidenceScore(score=0.7, divergence=None, tier="draft")  # type: ignore
        assert cs.accepts_at(0.5) is True
        assert cs.accepts_at(0.7) is True
        assert cs.accepts_at(0.8) is False

    def test_score_in_range(self, confidence_config: ConfidenceConfig) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = score_response(
            tier="draft",
            messages=messages,
            content="Hi there!",
            logprobs=None,
            config=confidence_config,
        )
        assert 0.0 <= result.score <= 1.0

    def test_easy_prompt_high_confidence(self, confidence_config: ConfidenceConfig) -> None:
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        result = score_response(
            tier="draft",
            messages=messages,
            content="The answer is 4.",
            logprobs=None,
            config=confidence_config,
        )
        assert result.score > 0.4

    def test_empty_response_low_confidence(self, confidence_config: ConfidenceConfig) -> None:
        messages = [{"role": "user", "content": "Tell me something important."}]
        result = score_response(
            tier="draft",
            messages=messages,
            content="",
            logprobs=None,
            config=confidence_config,
        )
        assert result.score < 0.5

    def test_logprobs_improve_high_confidence(self, confidence_config: ConfidenceConfig) -> None:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        content = "The answer is 4."
        result_with_lp = score_response(
            tier="draft",
            messages=messages,
            content=content,
            logprobs=[-0.05, -0.1],  # Very confident
            config=confidence_config,
        )
        result_without_lp = score_response(
            tier="draft",
            messages=messages,
            content=content,
            logprobs=[-8.0, -9.0],  # Very uncertain
            config=confidence_config,
        )
        assert result_with_lp.score > result_without_lp.score

    def test_tier_label_preserved(self, confidence_config: ConfidenceConfig) -> None:
        messages = [{"role": "user", "content": "test"}]
        for tier in ("draft", "qualifier", "target"):
            result = score_response(
                tier=tier,
                messages=messages,
                content="response",
                logprobs=None,
                config=confidence_config,
            )
            assert result.tier == tier
