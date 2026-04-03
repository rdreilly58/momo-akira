"""Verifier model — qualifier or target tier.

A verifier runs ONE forward pass over all K candidate tokens simultaneously
and returns logit distributions at each position.  The divergence calculator
then decides which tokens to accept.

Logit extraction:
  For a causal LM with input [ctx_0, ..., ctx_{n-1}, c_0, ..., c_{K-1}]:
    out.logits[0, n-1+t, :] = P(c_t | ctx_0, ..., ctx_{n-1}, c_0, ..., c_{t-1})
  So candidate logits are at slice [n_ctx-1 : n_ctx-1+K] — NOT at [n_ctx : n_ctx+K].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase  # type: ignore[import-untyped]

from momo_akira.config import ModelConfig


@dataclass
class VerifierOutput:
    logits: torch.Tensor  # shape (n_tokens, vocab_size) — raw pre-softmax logits
    # past_key_values retained for future KV-cache optimisation; not used by the engine.
    past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None


class VerifierModel:
    """Wraps a causal LM to act as a parallel token verifier.

    Unlike the draft model, this never generates tokens.  It only runs forward
    passes to obtain logit distributions so the engine can compute divergence.
    """

    def __init__(self, cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=self._resolve_dtype(cfg.dtype),
            device_map=cfg.device if cfg.device != "auto" else "auto",
            trust_remote_code=True,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def verify(
        self,
        candidate_ids: list[int],
        context_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,  # noqa: ARG002
    ) -> VerifierOutput:
        """Run a single forward pass over all candidate tokens at once.

        Always processes the full context + candidates in a single call.
        This avoids the off-by-one that arises when using a KV cache prefix:
        the logit for c_0 lives at position n_ctx-1 in the output, which is
        only visible when the context tokens are included in the input.

        Args:
            candidate_ids: The K token IDs proposed by the upstream tier.
            context_ids: The existing context preceding the candidates.
            past_key_values: Accepted but ignored — full context is always
                processed.  Kept for future KV-cache optimisation.

        Returns:
            VerifierOutput with logits[t] = the verifier's distribution used
            to verify candidate c_t, i.e., P(c_t | context, c_0, ..., c_{t-1}).
        """
        device = next(self.model.parameters()).device
        K = len(candidate_ids)
        ctx = context_ids.to(device)
        n_ctx = ctx.shape[1]

        candidate_tensor = torch.tensor([candidate_ids], device=device)  # (1, K)
        # Full sequence: (1, n_ctx + K)
        input_ids = torch.cat([ctx, candidate_tensor], dim=1)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False)

        # out.logits shape: (1, n_ctx + K, vocab_size)
        # For candidate c_t at sequence position n_ctx + t, the model's
        # prediction comes from output position n_ctx - 1 + t:
        #   t=0  → position n_ctx-1  (predicts c_0 given context)
        #   t=1  → position n_ctx    (predicts c_1 given context, c_0)
        #   ...
        candidate_logits = out.logits[0, n_ctx - 1 : n_ctx - 1 + K, :]  # (K, vocab)
        return VerifierOutput(logits=candidate_logits)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype, torch.float16)
