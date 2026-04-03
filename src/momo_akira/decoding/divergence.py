"""Divergence calculation and token acceptance logic.

The paper uses fuzzy speculative decoding (FSD) where:

    Div(P_A(x_t), P_B(x_t)) = |logit_A(x_t) - logit_B(x_t)|

This compares the raw logit values that each model assigns to the specific
token x_t that was chosen (not a full KL divergence).  Acceptance requires
Div ≤ τ (threshold).

Two acceptance variants from the paper:
- PSDF (Fuzzy): on rejection, resample from adjusted verifier distribution
- PSDA (Assisted): on rejection, take the next token directly from verifier
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AcceptanceResult:
    n_accepted: int           # how many tokens were accepted (0..K)
    accepted_ids: list[int]   # the accepted token IDs
    # If a rejection happened and we need a replacement token:
    replacement_id: int | None
    # Per-token divergence values for metrics/logging
    divergences: list[float]


class DivergenceChecker:
    """Computes Div() and applies threshold-based accept/reject per position."""

    def __init__(self, tau: float, variant: str = "psda") -> None:
        """
        Args:
            tau: Acceptance threshold.  Token is accepted when Div ≤ tau.
            variant: "psda" or "psdf".  Controls what happens on rejection.
        """
        self.tau = tau
        self.variant = variant

    def check(
        self,
        token_ids: list[int],
        proposer_logits: torch.Tensor,  # (K, vocab_size) — from draft/qualifier
        verifier_logits: torch.Tensor,  # (K, vocab_size) — from qualifier/target
    ) -> AcceptanceResult:
        """Apply the fuzzy acceptance rule to K candidate tokens.

        Processes tokens left-to-right and stops at the first rejection.
        On rejection:
          - PSDF: resample from adjusted verifier distribution
          - PSDA: take the token with the highest verifier logit (greedy from verifier)

        Returns:
            AcceptanceResult with n_accepted, accepted_ids, and optional replacement.
        """
        n = len(token_ids)
        assert proposer_logits.shape[0] == n
        assert verifier_logits.shape[0] == n

        accepted_ids: list[int] = []
        divergences: list[float] = []

        for t in range(n):
            tok = token_ids[t]
            div = float(
                torch.abs(proposer_logits[t, tok] - verifier_logits[t, tok]).item()
            )
            divergences.append(div)

            if div <= self.tau:
                accepted_ids.append(tok)
            else:
                # First rejection — compute replacement token
                replacement = self._replacement_token(
                    t, proposer_logits[t], verifier_logits[t]
                )
                return AcceptanceResult(
                    n_accepted=t,
                    accepted_ids=accepted_ids,
                    replacement_id=replacement,
                    divergences=divergences,
                )

        # All tokens accepted
        return AcceptanceResult(
            n_accepted=n,
            accepted_ids=accepted_ids,
            replacement_id=None,
            divergences=divergences,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _replacement_token(
        self,
        position: int,
        proposer_logits_t: torch.Tensor,  # (vocab_size,)
        verifier_logits_t: torch.Tensor,  # (vocab_size,)
    ) -> int:
        if self.variant == "psda":
            # PSDA (Assisted): take the verifier's greedy token
            return int(torch.argmax(verifier_logits_t).item())
        else:
            # PSDF (Fuzzy): resample from adjusted distribution
            # Standard SD resampling: max(0, P_verifier - P_proposer) normalised
            p_prop = torch.softmax(proposer_logits_t, dim=-1)
            p_ver = torch.softmax(verifier_logits_t, dim=-1)
            adjusted = torch.clamp(p_ver - p_prop, min=0.0)
            total = adjusted.sum()
            if total < 1e-10:
                # Fallback to greedy verifier if adjustment is near-zero
                return int(torch.argmax(verifier_logits_t).item())
            probs = adjusted / total
            return int(torch.multinomial(probs, num_samples=1).item())
