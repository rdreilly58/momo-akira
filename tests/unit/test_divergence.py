"""Tests for the divergence calculator and acceptance logic."""

import torch
import pytest

from momo_akira.decoding.divergence import DivergenceChecker, AcceptanceResult


VOCAB = 100


def _logits(token_vals: list[float], vocab: int = VOCAB) -> torch.Tensor:
    """Build a (K, vocab) logit tensor where each row has the given value at its token position."""
    n = len(token_vals)
    t = torch.zeros(n, vocab)
    for i, v in enumerate(token_vals):
        t[i, i] = v  # token i has logit at position i
    return t


def test_all_accept():
    # proposer and verifier agree perfectly
    checker = DivergenceChecker(tau=0.5, variant="psda")
    token_ids = [0, 1, 2]
    p_logits = _logits([1.0, 1.0, 1.0])
    v_logits = _logits([1.0, 1.0, 1.0])

    result = checker.check(token_ids, p_logits, v_logits)
    assert result.n_accepted == 3
    assert result.accepted_ids == [0, 1, 2]
    assert result.replacement_id is None


def test_reject_at_first():
    checker = DivergenceChecker(tau=0.1, variant="psda")
    token_ids = [0, 1, 2]
    p_logits = _logits([0.0, 1.0, 1.0])
    v_logits = _logits([5.0, 1.0, 1.0])  # large divergence at position 0

    result = checker.check(token_ids, p_logits, v_logits)
    assert result.n_accepted == 0
    assert result.accepted_ids == []
    assert result.replacement_id is not None  # verifier provides replacement


def test_reject_at_middle():
    checker = DivergenceChecker(tau=0.5, variant="psda")
    token_ids = [0, 1, 2]
    # position 0 and 1 agree, position 2 diverges
    p_logits = _logits([1.0, 1.0, 0.0])
    v_logits = _logits([1.0, 1.0, 9.0])  # div at 2

    result = checker.check(token_ids, p_logits, v_logits)
    assert result.n_accepted == 2
    assert result.accepted_ids == [0, 1]
    assert result.replacement_id is not None


def test_psda_replacement_is_greedy():
    """PSDA: replacement should be the verifier's argmax."""
    checker = DivergenceChecker(tau=0.0, variant="psda")
    token_ids = [0]
    # token 0's logit: proposer=0, verifier=5 → divergence=5 > tau=0 → reject
    p_logits = torch.zeros(1, VOCAB)
    v_logits = torch.zeros(1, VOCAB)
    p_logits[0, 0] = 0.0
    v_logits[0, 0] = 5.0   # large divergence at the chosen token (0)
    v_logits[0, 42] = 100.0  # verifier's preferred token is 42

    result = checker.check(token_ids, p_logits, v_logits)
    assert result.replacement_id == 42


def test_psdf_replacement_is_sampled():
    """PSDF: replacement should come from adjusted distribution (non-deterministic,
    just check it's a valid token index)."""
    checker = DivergenceChecker(tau=0.0, variant="psdf")
    token_ids = [0]
    # Force rejection by making divergence > 0
    p_logits = torch.zeros(1, VOCAB)
    v_logits = torch.zeros(1, VOCAB)
    p_logits[0, 0] = 0.0
    v_logits[0, 0] = 5.0   # divergence = 5 > tau = 0
    v_logits[0, 7] = 10.0

    result = checker.check(token_ids, p_logits, v_logits)
    assert result.replacement_id is not None
    assert 0 <= result.replacement_id < VOCAB


def test_divergences_recorded():
    checker = DivergenceChecker(tau=1.0, variant="psda")
    token_ids = [0, 1]
    p_logits = _logits([2.0, 3.0])
    v_logits = _logits([2.5, 3.5])

    result = checker.check(token_ids, p_logits, v_logits)
    assert len(result.divergences) == 2
    assert result.divergences[0] == pytest.approx(0.5, abs=1e-4)
    assert result.divergences[1] == pytest.approx(0.5, abs=1e-4)
