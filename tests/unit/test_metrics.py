"""Tests for metrics collection."""

from __future__ import annotations

import pytest

from momo_akira.cascade.pipeline import AcceptedTier, CascadeResult
from momo_akira.metrics import MetricsCollector
from momo_akira.models.base import LLMResponse


def _make_result(
    accepted_tier: AcceptedTier,
    draft_invoked: bool = True,
    qualifier_invoked: bool = False,
    target_invoked: bool = False,
    cost: float = 0.001,
    latency_ms: float = 500.0,
) -> CascadeResult:
    return CascadeResult(
        response=LLMResponse(
            content="test response",
            model_id="test-model",
            provider="mock",
            input_tokens=50,
            output_tokens=20,
            latency_ms=latency_ms,
            success=True,
        ),
        accepted_tier=accepted_tier,
        draft_invoked=draft_invoked,
        qualifier_invoked=qualifier_invoked,
        target_invoked=target_invoked,
        total_cost=cost,
        total_latency_ms=latency_ms,
        variant="psda",
    )


class TestMetricsCollector:
    def test_empty_metrics(self) -> None:
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["all_time"]["total_requests"] == 0

    def test_record_single_draft_acceptance(self) -> None:
        collector = MetricsCollector()
        result = _make_result(AcceptedTier.DRAFT, draft_invoked=True)
        collector.record(result)
        summary = collector.get_summary()
        assert summary["all_time"]["total_requests"] == 1

    def test_cost_accumulates(self) -> None:
        collector = MetricsCollector()
        for _ in range(3):
            collector.record(_make_result(AcceptedTier.DRAFT, cost=0.001))
        summary = collector.get_summary()
        assert abs(summary["all_time"]["total_cost_usd"] - 0.003) < 1e-9

    def test_multiple_tiers_tracked(self) -> None:
        collector = MetricsCollector()
        collector.record(_make_result(AcceptedTier.DRAFT))
        collector.record(_make_result(AcceptedTier.QUALIFIER, draft_invoked=True, qualifier_invoked=True))
        collector.record(
            _make_result(AcceptedTier.TARGET, draft_invoked=True, qualifier_invoked=True, target_invoked=True)
        )
        summary = collector.get_summary()
        assert summary["all_time"]["total_requests"] == 3

    def test_daily_metrics_separate(self) -> None:
        collector = MetricsCollector()
        collector.record(_make_result(AcceptedTier.DRAFT))
        today = collector.get_daily()
        assert today["total_requests"] == 1

    def test_all_time_accumulates_across_days(self) -> None:
        collector = MetricsCollector()
        for _ in range(5):
            collector.record(_make_result(AcceptedTier.DRAFT))
        all_time = collector.get_all_time()
        assert all_time["total_requests"] == 5
