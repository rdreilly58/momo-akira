"""Divergence measures adapted for API-based LLMs.

The paper uses Div(P_Q(x_t), P_D(x_t)) over logit distributions.
Since most LLM APIs don't expose token-level logits, we implement
response-level divergence proxies.

When logprobs ARE available (OpenAI supports this), we compute a
proper divergence approximation. Otherwise we fall back to
response-level heuristics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class DivergenceResult:
    """Result of a divergence computation."""

    # Normalized divergence in [0.0, 1.0] — higher = more divergent
    value: float
    # Which method was used
    method: str
    # Additional details for logging
    details: dict[str, float]

    @property
    def confidence(self) -> float:
        """Confidence = 1 - divergence."""
        return 1.0 - self.value


def logprob_divergence(logprobs: list[float]) -> DivergenceResult:
    """Compute divergence signal from per-token logprobs.

    Uses the mean negative log-probability as a proxy for model uncertainty.
    Lower average logprob (more negative) = higher uncertainty = higher divergence.

    Maps to [0, 1] using a sigmoid-like transform calibrated on typical
    LLM logprob ranges (-0.5 to -10 for common tokens).
    """
    if not logprobs:
        return DivergenceResult(value=0.5, method="logprob", details={})

    mean_lp = sum(logprobs) / len(logprobs)
    # Map [-10, 0] → [1, 0] using linear clamp
    # mean_lp near 0 = very confident = low divergence
    # mean_lp near -10 = very uncertain = high divergence
    normalized = max(0.0, min(1.0, -mean_lp / 10.0))
    return DivergenceResult(
        value=normalized,
        method="logprob",
        details={"mean_logprob": mean_lp, "n_tokens": len(logprobs)},
    )


def response_length_divergence(
    draft_content: str,
    expected_tokens: int | None = None,
) -> float:
    """Penalize responses that are extremely short (likely truncated/failed)."""
    words = len(draft_content.split())
    if words == 0:
        return 1.0  # Empty response = maximum divergence
    if words < 5:
        return 0.8
    if expected_tokens and expected_tokens > 100 and words < 10:
        return 0.7
    return 0.0


def response_coherence_score(content: str) -> float:
    """Heuristic coherence scoring — returns divergence value in [0, 1].

    Checks for signs of incoherence:
    - Truncated sentences
    - Repetition (likely degenerate output)
    - Broken formatting
    - Error/refusal phrases
    """
    if not content or not content.strip():
        return 1.0

    divergence = 0.0
    content_lower = content.lower().strip()

    # Check for error/refusal phrases
    error_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "error:", "exception:", "sorry, i", "as an ai",
    ]
    for phrase in error_phrases:
        if content_lower.startswith(phrase) or content_lower[:100].find(phrase) != -1:
            divergence += 0.3
            break

    # Check for severe repetition (degenerate output)
    words = content.split()
    if len(words) > 10:
        # Count unique bigrams vs total bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_ratio = len(set(bigrams)) / len(bigrams)
        if unique_ratio < 0.3:  # Very repetitive
            divergence += 0.5
        elif unique_ratio < 0.5:
            divergence += 0.2

    # Bonus for well-formed structure
    if content.strip().endswith((".", "!", "?", '"', "'")):
        divergence -= 0.05

    return max(0.0, min(1.0, divergence))


def prompt_complexity_score(messages: list[dict]) -> float:
    """Estimate prompt complexity as a divergence signal.

    Higher complexity = draft model less likely to produce high-quality output.
    Returns divergence value in [0, 1].

    Complexity indicators:
    - Multi-step reasoning requests
    - Technical/specialized vocabulary
    - Long context
    - Code generation requests
    - Math/logic problems
    """
    full_text = " ".join(
        str(msg.get("content", "")) for msg in messages
    ).lower()

    complexity = 0.0

    # Multi-step reasoning
    reasoning_markers = [
        "step by step", "step-by-step", "let's think", "reason through",
        "first", "then", "finally", "prove", "derive", "explain why",
        "analyze", "compare and contrast",
    ]
    for marker in reasoning_markers:
        if marker in full_text:
            complexity += 0.08
            if complexity >= 0.4:
                break

    # Code generation
    code_markers = [
        "write a function", "implement", "code", "algorithm",
        "debug", "fix the bug", "class ", "def ", "import ",
        "insert", "delete", "traverse", "search", "sort",
        "data structure", "linked list", "binary tree",
    ]
    for marker in code_markers:
        if marker in full_text:
            complexity += 0.12
            if complexity >= 0.5:
                break

    # Math / logic
    math_markers = [
        "calculate", "compute", "solve", "equation", "integral",
        "derivative", "probability", "statistics", "theorem",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",  # arithmetic expressions
    ]
    for marker in math_markers:
        if re.search(marker, full_text):
            complexity += 0.12
            if complexity >= 0.6:
                break

    # Context length signal (longer = harder)
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    if total_chars > 4000:
        complexity += 0.15
    elif total_chars > 2000:
        complexity += 0.08
    elif total_chars > 1000:
        complexity += 0.03

    return max(0.0, min(1.0, complexity))


def composite_divergence(
    messages: list[dict],
    draft_content: str,
    logprobs: list[float] | None,
    weights: dict[str, float],
    expected_tokens: int | None = None,
) -> DivergenceResult:
    """Compute composite divergence — the primary Div() proxy for API-based models.

    This replaces Div(P_Q(x_t), P_D(x_t)) from the paper with a weighted
    combination of available signals. The result maps to [0, 1] where 0 =
    identical distributions (accept draft) and 1 = maximal divergence (escalate).

    Args:
        messages: The input messages (for prompt complexity analysis)
        draft_content: The draft model's output text
        logprobs: Optional per-token logprobs from the draft model
        weights: Dict with keys: prompt_complexity, response_coherence, logprob
        expected_tokens: Optional expected output length hint
    """
    total_weight = sum(weights.values()) or 1.0

    components: dict[str, float] = {}
    weighted_sum = 0.0

    # 1. Response coherence signal
    coh_w = weights.get("response_coherence", 0.4)
    length_div = response_length_divergence(draft_content, expected_tokens)
    coherence_div = response_coherence_score(draft_content)
    # Blend length and coherence
    response_div = 0.4 * length_div + 0.6 * coherence_div
    components["response_coherence"] = response_div
    weighted_sum += coh_w * response_div

    # 2. Prompt complexity signal
    comp_w = weights.get("prompt_complexity", 0.3)
    prompt_div = prompt_complexity_score(messages)
    components["prompt_complexity"] = prompt_div
    weighted_sum += comp_w * prompt_div

    # 3. Logprob signal (if available)
    lp_w = weights.get("logprob", 0.3)
    if logprobs is not None and len(logprobs) > 0:
        lp_result = logprob_divergence(logprobs)
        lp_div = lp_result.value
    else:
        # Fall back: redistribute logprob weight to other signals proportionally
        if total_weight > 0:
            extra = lp_w / (total_weight - lp_w) if (total_weight - lp_w) > 0 else 0
            weighted_sum += (coh_w + comp_w) * extra * (
                coh_w * response_div + comp_w * prompt_div
            ) / max(coh_w + comp_w, 1e-9)
        lp_div = 0.0
        lp_w = 0.0
    components["logprob"] = lp_div
    weighted_sum += lp_w * lp_div

    effective_weight = coh_w + comp_w + lp_w
    final_divergence = weighted_sum / max(effective_weight, 1e-9)
    final_divergence = max(0.0, min(1.0, final_divergence))

    return DivergenceResult(
        value=final_divergence,
        method="composite",
        details=components,
    )
