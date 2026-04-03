"""Confidence scoring module.

Wraps the divergence module to produce a Confidence score C ∈ [0, 1]
where C = 1 - Divergence.

Acceptance criterion (adapted from paper):
  Stage 1: accept draft if C_draft >= tau_Q   (i.e., Div <= 1 - tau_Q)
  Stage 2: accept qualifier if C_qualifier >= tau_T
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ConfidenceConfig
from .divergence import DivergenceResult, composite_divergence


@dataclass
class ConfidenceScore:
    """Confidence assessment for a single model response."""

    # Overall confidence C ∈ [0, 1]
    score: float
    # Underlying divergence
    divergence: DivergenceResult
    # Tier that produced this response
    tier: str

    def accepts_at(self, threshold: float) -> bool:
        """Return True if this response is accepted at the given threshold."""
        return self.score >= threshold


def score_response(
    tier: str,
    messages: list[dict],
    content: str,
    logprobs: list[float] | None,
    config: ConfidenceConfig,
    expected_tokens: int | None = None,
) -> ConfidenceScore:
    """Compute confidence score for a model response.

    Args:
        tier: Which tier produced this response ("draft", "qualifier", "target")
        messages: Input messages (for prompt complexity)
        content: The model's text output
        logprobs: Optional token logprobs (OpenAI only)
        config: Confidence scoring configuration
        expected_tokens: Hint for expected output length

    Returns:
        ConfidenceScore with value in [0, 1]
    """
    weights = {
        "prompt_complexity": config.prompt_complexity_weight,
        "response_coherence": config.response_coherence_weight,
        "logprob": config.logprob_weight if (logprobs is not None) else 0.0,
    }

    divergence = composite_divergence(
        messages=messages,
        draft_content=content,
        logprobs=logprobs,
        weights=weights,
        expected_tokens=expected_tokens,
    )

    confidence = 1.0 - divergence.value
    return ConfidenceScore(score=confidence, divergence=divergence, tier=tier)
