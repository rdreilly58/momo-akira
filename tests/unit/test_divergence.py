"""Tests for divergence measures."""

from __future__ import annotations

import pytest

from openclaw_speculative_decoder.cascade.divergence import (
    composite_divergence,
    logprob_divergence,
    prompt_complexity_score,
    response_coherence_score,
    response_length_divergence,
)


class TestLogprobDivergence:
    def test_high_confidence_low_divergence(self) -> None:
        # logprobs near 0 = very confident = low divergence
        result = logprob_divergence([-0.1, -0.05, -0.02])
        assert result.value < 0.2

    def test_low_confidence_high_divergence(self) -> None:
        # logprobs near -10 = very uncertain = high divergence
        result = logprob_divergence([-9.0, -8.5, -9.5])
        assert result.value > 0.7

    def test_empty_logprobs_returns_neutral(self) -> None:
        result = logprob_divergence([])
        assert result.value == 0.5

    def test_confidence_is_complement(self) -> None:
        result = logprob_divergence([-2.0, -3.0])
        assert abs(result.confidence - (1.0 - result.value)) < 1e-9

    def test_method_label(self) -> None:
        result = logprob_divergence([-1.0])
        assert result.method == "logprob"


class TestResponseLengthDivergence:
    def test_empty_response_max_divergence(self) -> None:
        assert response_length_divergence("") == 1.0

    def test_very_short_response_high_divergence(self) -> None:
        assert response_length_divergence("ok") >= 0.7

    def test_normal_response_zero_divergence(self) -> None:
        long_content = "Paris is the capital of France and a major European city."
        assert response_length_divergence(long_content) == 0.0


class TestResponseCoherenceScore:
    def test_empty_content_max_divergence(self) -> None:
        assert response_coherence_score("") == 1.0

    def test_refusal_phrase_increases_divergence(self) -> None:
        content = "I cannot help with that request."
        score = response_coherence_score(content)
        assert score > 0.2

    def test_repetitive_output_high_divergence(self) -> None:
        repetitive = "the the the the the the the the the the the the the"
        score = response_coherence_score(repetitive)
        assert score > 0.4

    def test_good_response_low_divergence(self) -> None:
        content = "Paris is the capital of France. It is known for the Eiffel Tower."
        score = response_coherence_score(content)
        assert score < 0.3


class TestPromptComplexityScore:
    def test_simple_question_low_complexity(self) -> None:
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        score = prompt_complexity_score(messages)
        assert score < 0.3

    def test_code_request_high_complexity(self) -> None:
        messages = [{"role": "user", "content": "Implement a red-black tree with insert and delete."}]
        score = prompt_complexity_score(messages)
        assert score > 0.3

    def test_reasoning_request_higher_complexity(self) -> None:
        messages = [{"role": "user", "content": "Analyze step by step and prove why..."}]
        score = prompt_complexity_score(messages)
        assert score > 0.2

    def test_long_context_increases_complexity(self) -> None:
        long_content = "a " * 3000  # ~6000 chars
        messages = [{"role": "user", "content": long_content}]
        score = prompt_complexity_score(messages)
        assert score >= 0.1

    def test_score_in_range(self) -> None:
        for content in ["hi", "explain quantum computing", "write complex algorithm"]:
            messages = [{"role": "user", "content": content}]
            score = prompt_complexity_score(messages)
            assert 0.0 <= score <= 1.0


class TestCompositeDivergence:
    def test_easy_prompt_good_response_low_divergence(self) -> None:
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        content = "Paris is the capital of France."
        result = composite_divergence(
            messages=messages,
            draft_content=content,
            logprobs=None,
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        assert result.value < 0.5
        assert result.method == "composite"

    def test_empty_response_high_divergence(self) -> None:
        messages = [{"role": "user", "content": "Tell me something."}]
        result = composite_divergence(
            messages=messages,
            draft_content="",
            logprobs=None,
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        assert result.value > 0.5

    def test_logprobs_incorporated(self) -> None:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        content = "The answer is 4."
        # High confidence logprobs
        result_with_lp = composite_divergence(
            messages=messages,
            draft_content=content,
            logprobs=[-0.05, -0.1, -0.02],
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        # Low confidence logprobs
        result_low_lp = composite_divergence(
            messages=messages,
            draft_content=content,
            logprobs=[-8.0, -9.0, -8.5],
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        assert result_with_lp.value < result_low_lp.value

    def test_details_populated(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        result = composite_divergence(
            messages=messages,
            draft_content="Hello!",
            logprobs=None,
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        assert "response_coherence" in result.details
        assert "prompt_complexity" in result.details

    def test_result_in_range(self) -> None:
        messages = [{"role": "user", "content": "test"}]
        result = composite_divergence(
            messages=messages,
            draft_content="response",
            logprobs=None,
            weights={"prompt_complexity": 0.3, "response_coherence": 0.4, "logprob": 0.3},
        )
        assert 0.0 <= result.value <= 1.0
