"""PyramidSD decoding engine — the core of momo-akira v2.

Implements the two-stage nested speculative decoding loop from arxiv:2510.12966.

Outer loop (runs until max_new_tokens or EOS):
  Stage 1 — Draft → Qualifier (inner loop):
    Repeat until l_Q tokens accepted by qualifier:
      1. Draft generates l_D tokens (l_D fast forward passes)
      2. Qualifier verifies all l_D in ONE forward pass
      3. Accept/reject per-token via Div(logit_D, logit_Q) ≤ tau_q
      4. Accumulate accepted tokens in qualifier buffer

  Stage 2 — Qualifier buffer → Target:
    1. Target verifies all l_Q in ONE forward pass
    2. Accept/reject per-token via Div(logit_Q, logit_T) ≤ tau_t
    3. Accepted tokens are final output

Key insight (from paper):
  In Stage 2 the target compares against the *qualifier's* distributions,
  not the draft's.  P_Q is closer to P_T than P_D, so more tokens pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase  # type: ignore[import-untyped]

from momo_akira.config import Config
from momo_akira.decoding.adaptive import AdaptiveThresholdController
from momo_akira.decoding.cache import KVCacheManager
from momo_akira.decoding.divergence import DivergenceChecker
from momo_akira.models.draft import DraftModel
from momo_akira.models.verifier import VerifierModel


@dataclass
class GenerationMetrics:
    total_tokens: int = 0
    draft_proposed: int = 0
    draft_accepted: int = 0
    qualifier_proposed: int = 0
    qualifier_accepted: int = 0
    elapsed_s: float = 0.0
    tokens_per_second: float = 0.0
    stage1_iterations: int = 0   # inner loop iterations
    outer_iterations: int = 0

    @property
    def draft_acceptance_rate(self) -> float:
        if self.draft_proposed == 0:
            return 0.0
        return self.draft_accepted / self.draft_proposed

    @property
    def qualifier_acceptance_rate(self) -> float:
        if self.qualifier_proposed == 0:
            return 0.0
        return self.qualifier_accepted / self.qualifier_proposed

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "draft_acceptance_rate": round(self.draft_acceptance_rate, 3),
            "qualifier_acceptance_rate": round(self.qualifier_acceptance_rate, 3),
            "stage1_iterations": self.stage1_iterations,
            "outer_iterations": self.outer_iterations,
        }


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    metrics: GenerationMetrics
    adaptive_stats: dict[str, float] = field(default_factory=dict)


class PyramidSDEngine:
    """Orchestrates the full PyramidSD two-stage speculative decoding loop."""

    def __init__(
        self,
        draft: DraftModel,
        qualifier: VerifierModel,
        target: VerifierModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: Config,
    ) -> None:
        self.draft = draft
        self.qualifier = qualifier
        self.target = target
        self.tokenizer = tokenizer
        self.cfg = cfg

        self._draft_cache = KVCacheManager()
        self._qualifier_cache = KVCacheManager()
        self._target_cache = KVCacheManager()

        self._adaptive = AdaptiveThresholdController(cfg.decoding, cfg.adaptive)

        self._div_q = DivergenceChecker(cfg.decoding.tau_q, cfg.decoding.variant)
        self._div_t = DivergenceChecker(cfg.decoding.tau_t, cfg.decoding.variant)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a response for prompt using PyramidSD."""
        t0 = time.perf_counter()

        # Tokenize prompt
        device = next(self.draft.model.parameters()).device
        input_ids: torch.Tensor = self.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(device)

        # Reset caches for new request
        self._draft_cache.reset()
        self._qualifier_cache.reset()
        self._target_cache.reset()

        # Update divergence thresholds from adaptive controller
        self._div_q.tau = self._adaptive.tau_q
        self._div_t.tau = self._adaptive.tau_t

        output_ids: list[int] = []
        metrics = GenerationMetrics()

        eos_id = self.tokenizer.eos_token_id
        max_tokens = self.cfg.decoding.max_new_tokens
        l_d = self.cfg.decoding.speculative_length_d
        l_q = self.cfg.decoding.speculative_length_q

        # Full context = prompt + generated so far
        context = input_ids

        while len(output_ids) < max_tokens:
            metrics.outer_iterations += 1

            # ----------------------------------------------------------------
            # STAGE 1: Draft → Qualifier inner loop
            # Accumulate l_Q qualifier-accepted tokens
            # ----------------------------------------------------------------
            qualifier_buffer: list[int] = []
            # Logits from qualifier at each buffer position — needed for Stage 2
            qualifier_logits_buffer: list[torch.Tensor] = []  # list of (vocab_size,)

            # Track per outer-loop stats for adaptive controller
            loop_draft_proposed = 0
            loop_draft_accepted = 0

            while len(qualifier_buffer) < l_q:
                if len(output_ids) + len(qualifier_buffer) >= max_tokens:
                    break

                metrics.stage1_iterations += 1

                # 1a. Draft generates l_D tokens
                draft_result = self.draft.generate(
                    input_ids=context,
                    n_tokens=l_d,
                    past_key_values=self._draft_cache.cache,
                    temperature=self.cfg.decoding.temperature,
                )

                loop_draft_proposed += l_d
                metrics.draft_proposed += l_d

                # 1b. Qualifier verifies all l_D tokens in ONE forward pass
                qual_out = self.qualifier.verify(
                    candidate_ids=draft_result.token_ids,
                    context_ids=context,
                    past_key_values=self._qualifier_cache.cache,
                )

                # 1c. Accept/reject per-token via Div ≤ tau_q
                accept_result = self._div_q.check(
                    token_ids=draft_result.token_ids,
                    proposer_logits=draft_result.logits,
                    verifier_logits=qual_out.logits,
                )

                n_acc = accept_result.n_accepted
                loop_draft_accepted += n_acc
                metrics.draft_accepted += n_acc

                # Append accepted tokens to qualifier buffer
                for i, tok in enumerate(accept_result.accepted_ids):
                    qualifier_buffer.append(tok)
                    qualifier_logits_buffer.append(qual_out.logits[i])

                # If there was a rejection, add the replacement token from qualifier
                if accept_result.replacement_id is not None:
                    replacement = accept_result.replacement_id
                    qualifier_buffer.append(replacement)
                    # Use qualifier's logit at rejection position for Stage 2
                    qualifier_logits_buffer.append(
                        qual_out.logits[n_acc] if n_acc < qual_out.logits.shape[0]
                        else qual_out.logits[-1]
                    )
                    # Commit only the accepted tokens to draft cache
                    # (the replacement came from qualifier, not draft)
                    committed = accept_result.accepted_ids + [replacement]
                    self._draft_cache.commit(draft_result.past_key_values, len(committed))
                    self._qualifier_cache.commit(qual_out.past_key_values, len(committed))
                else:
                    # All l_D accepted — commit full batch
                    self._draft_cache.commit(draft_result.past_key_values, l_d)
                    self._qualifier_cache.commit(qual_out.past_key_values, l_d)

                # Update context for next inner-loop iteration
                # Append all tokens added to qualifier_buffer this batch
                n_added = n_acc + (1 if accept_result.replacement_id is not None else 0)
                if n_added > 0:
                    new_ids = torch.tensor(
                        [qualifier_buffer[-n_added:]],
                        device=device,
                    )
                    context = torch.cat([context, new_ids], dim=1)

                # Check for EOS in accepted tokens
                if eos_id is not None and any(
                    t == eos_id for t in accept_result.accepted_ids
                ):
                    break
                if accept_result.replacement_id == eos_id:
                    break

            # ----------------------------------------------------------------
            # STAGE 2: Qualifier buffer → Target
            # Target verifies all l_Q tokens in ONE forward pass
            # ----------------------------------------------------------------
            if not qualifier_buffer:
                break

            metrics.qualifier_proposed += len(qualifier_buffer)

            # Stack qualifier logits into (K, vocab) tensor
            qual_logits_tensor = torch.stack(qualifier_logits_buffer, dim=0)

            # Target verifies in ONE forward pass
            # Re-use target's cache from before the qualifier buffer was generated
            target_out = self.target.verify(
                candidate_ids=qualifier_buffer,
                context_ids=context[:, :-len(qualifier_buffer)],  # context before buffer
                past_key_values=self._target_cache.cache,
            )

            # Accept/reject via Div(logit_Q, logit_T) ≤ tau_t
            # Critical: compare target against qualifier (not draft)
            target_accept = self._div_t.check(
                token_ids=qualifier_buffer,
                proposer_logits=qual_logits_tensor,
                verifier_logits=target_out.logits,
            )

            final_tokens = target_accept.accepted_ids
            if target_accept.replacement_id is not None:
                final_tokens = final_tokens + [target_accept.replacement_id]

            metrics.qualifier_accepted += len(final_tokens)

            # Commit target cache
            self._target_cache.commit(target_out.past_key_values, len(final_tokens))

            # Append to output
            output_ids.extend(final_tokens)
            metrics.total_tokens += len(final_tokens)

            # Update adaptive controller
            self._adaptive.update(
                draft_accepted=loop_draft_accepted,
                draft_proposed=loop_draft_proposed,
                qualifier_accepted=len(final_tokens),
                qualifier_proposed=len(qualifier_buffer),
            )
            # Update divergence thresholds
            self._div_q.tau = self._adaptive.tau_q
            self._div_t.tau = self._adaptive.tau_t

            # Check EOS
            if eos_id is not None and eos_id in final_tokens:
                # Trim at EOS
                eos_pos = output_ids.index(eos_id, len(output_ids) - len(final_tokens))
                output_ids = output_ids[:eos_pos]
                break

        elapsed = time.perf_counter() - t0
        metrics.elapsed_s = elapsed
        metrics.tokens_per_second = metrics.total_tokens / elapsed if elapsed > 0 else 0.0

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            token_ids=output_ids,
            metrics=metrics,
            adaptive_stats=self._adaptive.stats,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> "PyramidSDEngine":
        """Load all three models and construct the engine."""
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        # Load shared tokenizer from the draft model
        # (all three must share the same vocabulary)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.models.draft.model_id, trust_remote_code=True
        )

        from momo_akira.models.draft import DraftModel
        from momo_akira.models.verifier import VerifierModel

        draft = DraftModel(cfg.models.draft, tokenizer)
        qualifier = VerifierModel(cfg.models.qualifier, tokenizer)
        target = VerifierModel(cfg.models.target, tokenizer)

        return cls(draft, qualifier, target, tokenizer, cfg)
