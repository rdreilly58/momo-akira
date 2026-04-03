"""KV cache manager.

Manages past_key_values for each model tier independently.  On token
acceptance, the cache is extended.  On rejection, it is rolled back to the
last committed position so the next draft batch starts from clean state.

HuggingFace stores past_key_values as:
    tuple of (num_layers) tuples of (key, value) tensors, each shape
    (batch_size, num_heads, seq_len, head_dim).

We track a "committed length" separately from the tensor's actual seq_len so
we can detect how many tokens were appended by a speculative batch that needs
rollback.
"""

from __future__ import annotations

import torch


class KVCacheManager:
    """Manages KV cache state for one model tier."""

    def __init__(self) -> None:
        self._cache: tuple[tuple[torch.Tensor, ...], ...] | None = None
        self._committed_len: int = 0  # number of tokens whose cache is authoritative

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache(self) -> tuple[tuple[torch.Tensor, ...], ...] | None:
        return self._cache

    @property
    def committed_len(self) -> int:
        return self._committed_len

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def commit(
        self,
        new_cache: tuple[tuple[torch.Tensor, ...], ...],
        n_new_tokens: int,
    ) -> None:
        """Accept new_cache as authoritative.  Call after a successful accept."""
        self._cache = new_cache
        self._committed_len += n_new_tokens

    def rollback(self) -> None:
        """Roll back any speculative tokens, keeping only committed state.

        After rollback the cache represents exactly committed_len tokens.
        If the cache was extended speculatively we trim it back.
        """
        if self._cache is None:
            return

        target_len = self._committed_len
        trimmed = self._trim_cache(self._cache, target_len)
        self._cache = trimmed

    def reset(self) -> None:
        """Full reset — used at the start of each new request."""
        self._cache = None
        self._committed_len = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_cache(
        cache: tuple[tuple[torch.Tensor, ...], ...],
        target_seq_len: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Trim each layer's key/value tensors to target_seq_len along dim 2."""
        trimmed_layers = []
        for layer in cache:
            trimmed_kv = tuple(t[:, :, :target_seq_len, :] for t in layer)
            trimmed_layers.append(trimmed_kv)
        return tuple(trimmed_layers)

    def current_seq_len(self) -> int:
        """Return the actual seq_len currently stored in the cache tensors."""
        if self._cache is None:
            return 0
        first_layer_key = self._cache[0][0]
        return int(first_layer_key.shape[2])
