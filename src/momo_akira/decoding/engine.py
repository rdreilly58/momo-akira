"""PyramidSD decoding engine — the core of momo-akira v2.

Implements the two-stage nested speculative decoding loop from arxiv:2510.12966.

Outer loop (runs until max_new_tokens or EOS):
  Stage 1 — Draft → Qualifier (inner loop):
    Repeat until l_Q tokens accepted by qualifier:
      1. Draft generates l_D tokens (l_D sequential forward passes, fast)
      2. Qualifier verifies all l_D in ONE forward pass
      3. Accept/reject per-token via Div(logit_D, logit_Q) ≤ tau_q
      4. Accumulate accepted tokens (+ one replacement on rejection) in buffer

  Stage 2 — Qualifier buffer → Target:
    1. Target verifies all l_Q buffer tokens in ONE forward pass
    2. Accept/reject per-token via Div(logit_Q, logit_T) ≤ tau_t
    3. Accepted tokens are final output

Key insight from the paper (§3):
  In Stage 2 the target compares against the *qualifier's* distributions,
  not the draft's.  P_Q is closer to P_T than P_D, so more tokens pass.

Note on KV caching:
  For correctness the verifiers always process the full context + candidates.
  The logit for candidate c_0 lives at output position n_ctx-1, which is only
  present when the full context is included in the input.  Incremental KV-cache
  verification would require storing the "pending" logit from the previous call
  and is left as a future optimisation.  The draft model also foregoes caching
  to keep the context bookkeeping simple and correct.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase  # type: ignore[import-untyped]

from momo_akira.config import Config
from momo_akira.decoding.adaptive import AdaptiveThresholdController
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

        self._adaptive = AdaptiveThresholdController(cfg.decoding, cfg.adaptive)
        self._div_q = DivergenceChecker(cfg.decoding.tau_q, cfg.decoding.variant)
        self._div_t = DivergenceChecker(cfg.decoding.tau_t, cfg.decoding.variant)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a response for prompt using PyramidSD."""
        t0 = time.perf_counter()

        device = next(self.draft.model.parameters()).device
        input_ids: torch.Tensor = self.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(device)

        # Sync divergence thresholds with the adaptive controller's current values.
        self._div_q.tau = self._adaptive.tau_q
        self._div_t.tau = self._adaptive.tau_t

        output_ids: list[int] = []
        metrics = GenerationMetrics()

        eos_id = self.tokenizer.eos_token_id
        max_tokens = self.cfg.decoding.max_new_tokens
        l_d = self.cfg.decoding.speculative_length_d
        l_q = self.cfg.decoding.speculative_length_q

        # Full context (prompt + all accepted output so far) grows each outer iteration.
        context = input_ids

        while len(output_ids) < max_tokens:
            metrics.outer_iterations += 1

            # Snapshot context before Stage 1.
            # After Stage 2 we reset context to context_s1 + final_tokens,
            # discarding any qualifier_buffer tokens the target rejected.
            context_s1 = context

            # ----------------------------------------------------------------
            # STAGE 1: Draft → Qualifier inner loop
            # Accumulate l_Q qualifier-accepted tokens in qualifier_buffer.
            # ----------------------------------------------------------------
            qualifier_buffer: list[int] = []
            # qualifier_logits_buffer[t] = the qualifier's logit distribution
            # at the position where qualifier_buffer[t] was chosen.
            # Stage 2 uses these so the target compares against P_Q, not P_D.
            qualifier_logits_buffer: list[torch.Tensor] = []

            loop_draft_proposed = 0
            loop_draft_accepted = 0
            eos_in_stage1 = False

            while len(qualifier_buffer) < l_q:
                if len(output_ids) + len(qualifier_buffer) >= max_tokens:
                    break

                metrics.stage1_iterations += 1

                # 1a. Draft generates l_D tokens sequentially.
                #     No KV cache: always process the full current context so
                #     the sequence length in the cache stays in sync with
                #     context.shape[1] — avoids off-by-N mismatches on rollback.
                draft_result = self.draft.generate(
                    input_ids=context,
                    n_tokens=l_d,
                    past_key_values=None,
                    temperature=self.cfg.decoding.temperature,
                )

                loop_draft_proposed += l_d
                metrics.draft_proposed += l_d

                # 1b. Qualifier verifies all l_D tokens in ONE forward pass.
                qual_out = self.qualifier.verify(
                    candidate_ids=draft_result.token_ids,
                    context_ids=context,
                )

                # 1c. Accept/reject per-token via Div(logit_D, logit_Q) ≤ tau_q.
                accept_result = self._div_q.check(
                    token_ids=draft_result.token_ids,
                    proposer_logits=draft_result.logits,
                    verifier_logits=qual_out.logits,
                )

                n_acc = accept_result.n_accepted
                loop_draft_accepted += n_acc
                metrics.draft_accepted += n_acc

                # 1d. Collect accepted tokens and their qualifier logit distributions.
                for i, tok in enumerate(accept_result.accepted_ids):
                    qualifier_buffer.append(tok)
                    qualifier_logits_buffer.append(qual_out.logits[i])

                # 1e. On rejection: add one replacement token from qualifier.
                if accept_result.replacement_id is not None:
                    replacement = accept_result.replacement_id
                    qualifier_buffer.append(replacement)
                    # Qualifier's logit at the rejection position.
                    repl_idx = min(n_acc, qual_out.logits.shape[0] - 1)
                    qualifier_logits_buffer.append(qual_out.logits[repl_idx])

                # 1f. Extend context with exactly the tokens added to the buffer.
                n_added = n_acc + (1 if accept_result.replacement_id is not None else 0)
                if n_added > 0:
                    new_ids = torch.tensor(
                        [qualifier_buffer[-n_added:]],
                        device=device,
                    )
                    context = torch.cat([context, new_ids], dim=1)

                # 1g. Stop Stage 1 early if EOS appeared.
                new_tokens_this_batch = accept_result.accepted_ids + (
                    [accept_result.replacement_id]
                    if accept_result.replacement_id is not None
                    else []
                )
                if eos_id is not None and eos_id in new_tokens_this_batch:
                    eos_in_stage1 = True
                    break

            # ----------------------------------------------------------------
            # STAGE 2: Qualifier buffer → Target
            # Target verifies all l_Q buffer tokens in ONE forward pass.
            # ----------------------------------------------------------------
            if not qualifier_buffer:
                break

            metrics.qualifier_proposed += len(qualifier_buffer)

            # Stack qualifier logit distributions into (K, vocab) tensor.
            qual_logits_tensor = torch.stack(qualifier_logits_buffer, dim=0)

            # Target processes: [context_s1, q_0, q_1, ..., q_{K-1}]
            # The target's logit at position n_ctx_s1 - 1 + t predicts q_t
            # given context_s1 + q_0, ..., q_{t-1}.  This matches exactly
            # what the qualifier computed for each buffer token.
            target_out = self.target.verify(
                candidate_ids=qualifier_buffer,
                context_ids=context_s1,
            )

            # Accept/reject via Div(logit_Q, logit_T) ≤ tau_t.
            # Critical: compare target against qualifier (not draft).
            target_accept = self._div_t.check(
                token_ids=qualifier_buffer,
                proposer_logits=qual_logits_tensor,
                verifier_logits=target_out.logits,
            )

            final_tokens = list(target_accept.accepted_ids)
            if target_accept.replacement_id is not None:
                final_tokens.append(target_accept.replacement_id)

            metrics.qualifier_accepted += len(final_tokens)
            output_ids.extend(final_tokens)
            metrics.total_tokens += len(final_tokens)

            # Reset context to context_s1 + final_tokens only.
            # This discards any qualifier_buffer tokens that the target rejected.
            if final_tokens:
                final_tensor = torch.tensor([final_tokens], device=device)
                context = torch.cat([context_s1, final_tensor], dim=1)
            else:
                context = context_s1

            # Update adaptive controller and re-sync thresholds.
            self._adaptive.update(
                draft_accepted=loop_draft_accepted,
                draft_proposed=loop_draft_proposed,
                qualifier_accepted=len(final_tokens),
                qualifier_proposed=len(qualifier_buffer),
            )
            self._div_q.tau = self._adaptive.tau_q
            self._div_t.tau = self._adaptive.tau_t

            # Terminate on EOS.
            if eos_id is not None and eos_id in final_tokens:
                eos_pos = output_ids.index(eos_id, len(output_ids) - len(final_tokens))
                output_ids = output_ids[:eos_pos]
                break

            if eos_in_stage1 and not final_tokens:
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
        """Load all three models sequentially to minimize peak memory."""
        import gc

        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        # Load shared tokenizer from the draft model
        # (all three must share the same vocabulary).
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.models.draft.model_id, trust_remote_code=True
        )

        from momo_akira.models.draft import DraftModel
        from momo_akira.models.verifier import VerifierModel

        # Load sequentially with GC between each to minimize peak memory
        print("[momo-akira] Loading draft model...")
        draft = DraftModel(cfg.models.draft, tokenizer)
        gc.collect()

        print("[momo-akira] Loading qualifier model...")
        qualifier = VerifierModel(cfg.models.qualifier, tokenizer)
        gc.collect()

        print("[momo-akira] Loading target model...")
        target = VerifierModel(cfg.models.target, tokenizer)
        gc.collect()

        print("[momo-akira] All models loaded. Engine ready.")
        return cls(draft, qualifier, target, tokenizer, cfg)
