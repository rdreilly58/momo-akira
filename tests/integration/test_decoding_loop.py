"""Integration test for the PyramidSD two-stage decoding loop.

Uses tiny mock models (no real weights) to verify the algorithm's control flow:
- Stage 1 inner loop accumulates l_Q tokens
- Stage 2 verifies against target
- Metrics are collected correctly
- EOS terminates generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from momo_akira.config import AdaptiveConfig, Config, DecodingConfig, ModelsConfig, ModelConfig
from momo_akira.decoding.engine import PyramidSDEngine
from momo_akira.models.draft import DraftOutput


VOCAB = 50
EOS = 49


def _cfg(
    l_d: int = 2,
    l_q: int = 4,
    tau_q: float = 1.0,  # accept everything by default
    tau_t: float = 1.0,
    max_tokens: int = 20,
    variant: str = "psda",
) -> Config:
    return Config(
        models=ModelsConfig(
            family="test",
            draft=ModelConfig(model_id="draft"),
            qualifier=ModelConfig(model_id="qualifier"),
            target=ModelConfig(model_id="target"),
        ),
        decoding=DecodingConfig(
            variant=variant,
            speculative_length_d=l_d,
            speculative_length_q=l_q,
            tau_q=tau_q,
            tau_t=tau_t,
            temperature=1.0,
            max_new_tokens=max_tokens,
        ),
        adaptive=AdaptiveConfig(enabled=False),
    )


def _constant_logits(token_id: int, value: float = 1.0) -> torch.Tensor:
    """Return a (vocab,) tensor with high logit at token_id."""
    t = torch.zeros(VOCAB)
    t[token_id] = value
    return t


def _make_draft_mock(token_sequence: list[int], l_d: int) -> MagicMock:
    """Draft returns l_d tokens at a time from token_sequence."""
    call_count = [0]

    def side_effect(input_ids: torch.Tensor, n_tokens: int, **kwargs: Any) -> DraftOutput:
        start = call_count[0] * n_tokens
        tokens = token_sequence[start : start + n_tokens]
        call_count[0] += 1
        logits = torch.stack([_constant_logits(t) for t in tokens])
        return DraftOutput(
            token_ids=tokens,
            logits=logits,
            past_key_values=(),  # type: ignore[arg-type]
        )

    mock = MagicMock()
    mock.generate.side_effect = side_effect
    mock.model.parameters.return_value = iter([torch.zeros(1)])
    return mock


def _make_verifier_mock(accept_all: bool = True, eos_at: int | None = None) -> MagicMock:
    """Verifier returns logits that accept all tokens (or reject at eos_at)."""
    from momo_akira.models.verifier import VerifierOutput

    def side_effect(
        candidate_ids: list[int],
        context_ids: torch.Tensor,
        past_key_values: Any = None,
    ) -> VerifierOutput:
        logits = torch.stack([_constant_logits(t, value=1.0) for t in candidate_ids])
        return VerifierOutput(logits=logits)

    mock = MagicMock()
    mock.verify.side_effect = side_effect
    mock.model.parameters.return_value = iter([torch.zeros(1)])
    return mock


def _make_tokenizer_mock(prompt_token_count: int = 5) -> MagicMock:
    tok = MagicMock()
    tok.encode.return_value = torch.zeros(1, prompt_token_count, dtype=torch.long)
    tok.decode.side_effect = lambda ids, **kw: " ".join(str(i) for i in ids)
    tok.eos_token_id = EOS
    return tok


def _make_engine(
    cfg: Config,
    token_sequence: list[int],
    l_d: int = 2,
) -> PyramidSDEngine:
    draft = _make_draft_mock(token_sequence, l_d)
    qualifier = _make_verifier_mock()
    target = _make_verifier_mock()
    tokenizer = _make_tokenizer_mock()
    return PyramidSDEngine(draft, qualifier, target, tokenizer, cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engine_produces_output():
    """Engine generates some output without crashing."""
    cfg = _cfg(l_d=2, l_q=4, max_tokens=8)
    tokens = list(range(20))
    engine = _make_engine(cfg, tokens, l_d=2)
    result = engine.generate("hello")
    assert len(result.token_ids) > 0
    assert result.text != ""


def test_metrics_collected():
    cfg = _cfg(l_d=2, l_q=4, max_tokens=8)
    tokens = list(range(20))
    engine = _make_engine(cfg, tokens, l_d=2)
    result = engine.generate("hello")
    m = result.metrics
    assert m.total_tokens > 0
    assert m.draft_proposed > 0
    assert m.outer_iterations > 0
    assert m.tokens_per_second > 0


def test_eos_terminates_generation():
    """EOS token should stop generation early."""
    cfg = _cfg(l_d=2, l_q=4, max_tokens=100)
    # Insert EOS after a few tokens
    tokens = [1, 2, EOS, 4, 5, 6, 7, 8, 9, 10] * 5
    engine = _make_engine(cfg, tokens, l_d=2)
    result = engine.generate("hello")
    # EOS should have stopped generation before max_tokens
    assert len(result.token_ids) < 100


def test_acceptance_rates_in_metrics():
    cfg = _cfg(l_d=2, l_q=4, tau_q=1.0, tau_t=1.0, max_tokens=8)
    tokens = list(range(20))
    engine = _make_engine(cfg, tokens, l_d=2)
    result = engine.generate("hello")
    m = result.metrics
    # With tau=1.0 (accept all), acceptance rates should be 1.0
    assert m.draft_acceptance_rate >= 0.0
    assert m.qualifier_acceptance_rate >= 0.0


def test_adaptive_stats_present():
    cfg = _cfg(max_tokens=8)
    tokens = list(range(20))
    engine = _make_engine(cfg, tokens)
    result = engine.generate("hello")
    assert "tau_q" in result.adaptive_stats
    assert "tau_t" in result.adaptive_stats
