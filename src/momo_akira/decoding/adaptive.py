"""Adaptive threshold controller.

Monitors per-stage acceptance rates (beta_Q, beta_T) using exponential
moving averages and nudges tau_q / tau_t to hit the target rates.

If acceptance is below target → lower tau (be more permissive).
If acceptance is above target → raise tau (be stricter).
"""

from __future__ import annotations

from momo_akira.config import AdaptiveConfig, DecodingConfig


class AdaptiveThresholdController:
    def __init__(self, decoding: DecodingConfig, adaptive: AdaptiveConfig) -> None:
        self.cfg = adaptive
        self.tau_q = decoding.tau_q
        self.tau_t = decoding.tau_t
        self._alpha = adaptive.learning_rate

        # EMA of acceptance rates
        self._ema_q: float | None = None
        self._ema_t: float | None = None
        self._tokens_seen: int = 0

        # Soft bounds — keep thresholds sane
        self._tau_q_min = 0.05
        self._tau_q_max = 1.0
        self._tau_t_min = 0.05
        self._tau_t_max = 1.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        draft_accepted: int,
        draft_proposed: int,
        qualifier_accepted: int,
        qualifier_proposed: int,
    ) -> None:
        """Call after each outer-loop iteration to update thresholds."""
        if not self.cfg.enabled:
            return

        self._tokens_seen += draft_proposed + qualifier_proposed

        if draft_proposed > 0:
            rate_q = draft_accepted / draft_proposed
            self._ema_q = self._ema_update(self._ema_q, rate_q)

        if qualifier_proposed > 0:
            rate_t = qualifier_accepted / qualifier_proposed
            self._ema_t = self._ema_update(self._ema_t, rate_t)

        if self._tokens_seen < self.cfg.warmup_tokens:
            return

        if self._ema_q is not None:
            self.tau_q = self._adjust(
                self.tau_q,
                self._ema_q,
                self.cfg.target_draft_acceptance,
                self._tau_q_min,
                self._tau_q_max,
            )

        if self._ema_t is not None:
            self.tau_t = self._adjust(
                self.tau_t,
                self._ema_t,
                self.cfg.target_qualifier_acceptance,
                self._tau_t_min,
                self._tau_t_max,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ema_update(self, current: float | None, new_value: float) -> float:
        if current is None:
            return new_value
        return self._alpha * new_value + (1 - self._alpha) * current

    def _adjust(
        self,
        tau: float,
        measured: float,
        target: float,
        lo: float,
        hi: float,
    ) -> float:
        """If measured < target (too strict), lower tau.  If measured > target, raise it."""
        error = measured - target
        # Move tau in the direction that corrects the error
        # Higher tau → stricter → lower acceptance, so:
        #   measured < target → lower tau (−= step)
        #   measured > target → raise tau (+= step)
        step = self._alpha * error
        new_tau = tau + step
        return max(lo, min(hi, new_tau))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, float]:
        return {
            "tau_q": self.tau_q,
            "tau_t": self.tau_t,
            "ema_acceptance_q": self._ema_q if self._ema_q is not None else 0.0,
            "ema_acceptance_t": self._ema_t if self._ema_t is not None else 0.0,
            "tokens_seen": float(self._tokens_seen),
        }
