"""Cost tracking and metrics aggregation.

Tracks per-request and daily aggregated metrics:
  - Request counts by tier
  - Total cost (USD) by tier and overall
  - Acceptance rates by tier
  - Average latency by tier
  - Token counts
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .cascade.pipeline import AcceptedTier, CascadeResult


@dataclass
class TierMetrics:
    requests: int = 0
    accepted: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.accepted / self.requests

    @property
    def avg_latency_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.total_latency_ms / self.requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass
class DailyMetrics:
    date: str = ""
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    tiers: dict[str, TierMetrics] = field(
        default_factory=lambda: {
            "draft": TierMetrics(),
            "qualifier": TierMetrics(),
            "target": TierMetrics(),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.total_latency_ms / max(self.total_requests, 1), 1),
            "tiers": {name: metrics.to_dict() for name, metrics in self.tiers.items()},
        }


class MetricsCollector:
    """Thread-safe metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._daily: dict[str, DailyMetrics] = {}
        self._all_time = DailyMetrics(date="all_time")

    def record(self, result: CascadeResult) -> None:
        """Record a completed cascade result."""
        today = date.today().isoformat()
        with self._lock:
            if today not in self._daily:
                self._daily[today] = DailyMetrics(date=today)
            daily = self._daily[today]
            self._update(daily, result)
            self._update(self._all_time, result)

    def _update(self, metrics: DailyMetrics, result: CascadeResult) -> None:
        metrics.total_requests += 1
        metrics.total_cost_usd += result.total_cost
        metrics.total_latency_ms += result.total_latency_ms

        # Track which tiers were invoked and which was accepted
        for tier_name, invoked in [
            ("draft", result.draft_invoked),
            ("qualifier", result.qualifier_invoked),
            ("target", result.target_invoked),
        ]:
            if invoked:
                tm = metrics.tiers[tier_name]
                tm.requests += 1
                if result.accepted_tier.value == tier_name:
                    tm.accepted += 1
                tm.total_cost_usd += result.total_cost  # approximate — full cost on accepted tier
                tm.total_latency_ms += result.response.latency_ms

        # Token counts on the final accepted response
        accepted = metrics.tiers[result.accepted_tier.value]
        accepted.total_input_tokens += result.response.input_tokens
        accepted.total_output_tokens += result.response.output_tokens

    def get_daily(self, day: str | None = None) -> dict[str, Any]:
        today = day or date.today().isoformat()
        with self._lock:
            if today not in self._daily:
                return DailyMetrics(date=today).to_dict()
            return self._daily[today].to_dict()

    def get_all_time(self) -> dict[str, Any]:
        with self._lock:
            return self._all_time.to_dict()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "all_time": self._all_time.to_dict(),
                "today": self._daily.get(date.today().isoformat(), DailyMetrics(date=date.today().isoformat())).to_dict(),
                "days_with_data": sorted(self._daily.keys()),
            }
