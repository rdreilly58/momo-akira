"""Adaptive threshold controller.

Tunes tau_Q and tau_T at runtime based on measured acceptance rates.
Uses exponential moving average (EMA) to track acceptance rates and
adjusts thresholds toward target rates.

Paper insight: optimal tau_Q and tau_T vary by task — runtime adaptation
helps find the right operating point automatically.

Control law:
  If actual_acceptance_rate > target:
    Increase threshold (be stricter — let fewer pass)
  If actual_acceptance_rate < target:
    Decrease threshold (be more lenient — let more pass)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..config import AdaptiveThresholdConfig


@dataclass
class TierStats:
    """Acceptance rate statistics for one tier."""

    total_requests: int = 0
    accepted: int = 0
    # EMA of acceptance rate
    ema_acceptance: float = 0.5

    @property
    def raw_acceptance_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.accepted / self.total_requests


@dataclass
class ThresholdState:
    """Current threshold values."""

    tau_q: float
    tau_t: float
    draft_stats: TierStats = field(default_factory=TierStats)
    qualifier_stats: TierStats = field(default_factory=TierStats)
    total_updates: int = 0


class AdaptiveThresholdController:
    """Controls tau_Q and tau_T based on observed acceptance rates.

    Thread-safe; designed to be shared across concurrent requests.
    """

    def __init__(self, config: AdaptiveThresholdConfig, initial_tau_q: float, initial_tau_t: float) -> None:
        self._config = config
        self._state = ThresholdState(tau_q=initial_tau_q, tau_t=initial_tau_t)
        self._lock = threading.Lock()

    @property
    def tau_q(self) -> float:
        return self._state.tau_q

    @property
    def tau_t(self) -> float:
        return self._state.tau_t

    def record_draft(self, accepted: bool) -> None:
        """Record whether the draft tier's response was accepted."""
        if not self._config.enabled:
            return
        with self._lock:
            stats = self._state.draft_stats
            stats.total_requests += 1
            if accepted:
                stats.accepted += 1
            alpha = self._config.learning_rate
            stats.ema_acceptance = (1 - alpha) * stats.ema_acceptance + alpha * (1.0 if accepted else 0.0)
            self._maybe_update()

    def record_qualifier(self, accepted: bool) -> None:
        """Record whether the qualifier tier's response was accepted."""
        if not self._config.enabled:
            return
        with self._lock:
            stats = self._state.qualifier_stats
            stats.total_requests += 1
            if accepted:
                stats.accepted += 1
            alpha = self._config.learning_rate
            stats.ema_acceptance = (1 - alpha) * stats.ema_acceptance + alpha * (1.0 if accepted else 0.0)
            self._maybe_update()

    def _maybe_update(self) -> None:
        """Update thresholds if we have enough data (called with lock held)."""
        total = self._state.draft_stats.total_requests
        if total < self._config.warmup_requests:
            return

        self._update_tau_q()
        self._update_tau_t()
        self._enforce_ordering()
        self._state.total_updates += 1

    def _update_tau_q(self) -> None:
        """Adjust tau_Q based on draft acceptance rate."""
        cfg = self._config
        stats = self._state.draft_stats
        error = stats.ema_acceptance - cfg.target_draft_acceptance
        # Proportional adjustment
        delta = cfg.learning_rate * error * 0.1
        new_tau_q = self._state.tau_q + delta
        self._state.tau_q = max(cfg.tau_q_min, min(cfg.tau_q_max, new_tau_q))

    def _update_tau_t(self) -> None:
        """Adjust tau_T based on qualifier acceptance rate."""
        cfg = self._config
        stats = self._state.qualifier_stats
        if stats.total_requests == 0:
            return
        error = stats.ema_acceptance - cfg.target_qualifier_acceptance
        delta = cfg.learning_rate * error * 0.1
        new_tau_t = self._state.tau_t + delta
        self._state.tau_t = max(cfg.tau_t_min, min(cfg.tau_t_max, new_tau_t))

    def _enforce_ordering(self) -> None:
        """Ensure tau_Q <= tau_T (paper recommendation)."""
        if self._state.tau_q > self._state.tau_t:
            # Average them and enforce ordering
            mid = (self._state.tau_q + self._state.tau_t) / 2.0
            self._state.tau_q = mid - 0.01
            self._state.tau_t = mid + 0.01

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "tau_q": self._state.tau_q,
                "tau_t": self._state.tau_t,
                "draft_acceptance_rate": self._state.draft_stats.ema_acceptance,
                "qualifier_acceptance_rate": self._state.qualifier_stats.ema_acceptance,
                "total_requests": self._state.draft_stats.total_requests,
                "total_threshold_updates": self._state.total_updates,
            }
