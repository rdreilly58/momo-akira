"""Tests for the KV cache manager."""

import torch
import pytest

from momo_akira.decoding.cache import KVCacheManager


def _make_cache(seq_len: int, n_layers: int = 2, n_heads: int = 4, head_dim: int = 8) -> tuple:
    """Build a fake past_key_values tuple."""
    return tuple(
        (
            torch.zeros(1, n_heads, seq_len, head_dim),
            torch.zeros(1, n_heads, seq_len, head_dim),
        )
        for _ in range(n_layers)
    )


def test_initial_state():
    m = KVCacheManager()
    assert m.cache is None
    assert m.committed_len == 0
    assert m.current_seq_len() == 0


def test_commit_updates_len():
    m = KVCacheManager()
    cache = _make_cache(5)
    m.commit(cache, n_new_tokens=5)
    assert m.committed_len == 5
    assert m.current_seq_len() == 5


def test_commit_accumulates():
    m = KVCacheManager()
    m.commit(_make_cache(5), n_new_tokens=5)
    m.commit(_make_cache(8), n_new_tokens=3)
    assert m.committed_len == 8


def test_rollback_trims_to_committed():
    m = KVCacheManager()
    m.commit(_make_cache(10), n_new_tokens=10)
    # Speculatively extend to 15 (without committing)
    m._cache = _make_cache(15)
    assert m.current_seq_len() == 15

    m.rollback()
    assert m.current_seq_len() == 10
    assert m.committed_len == 10


def test_rollback_no_op_if_nothing_speculative():
    m = KVCacheManager()
    m.commit(_make_cache(5), n_new_tokens=5)
    m.rollback()
    assert m.current_seq_len() == 5


def test_reset_clears_everything():
    m = KVCacheManager()
    m.commit(_make_cache(10), n_new_tokens=10)
    m.reset()
    assert m.cache is None
    assert m.committed_len == 0
    assert m.current_seq_len() == 0
