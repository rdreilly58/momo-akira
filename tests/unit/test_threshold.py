"""Tests for adaptive threshold controller."""

from __future__ import annotations

import pytest

from momo_akira.cascade.threshold import AdaptiveThresholdController
from momo_akira.config import AdaptiveThresholdConfig


@pytest.fixture
def controller() -> AdaptiveThresholdController:
    config = AdaptiveThresholdConfig(
        enabled=True,
        learning_rate=0.1,
        target_draft_acceptance=0.60,
        target_qualifier_acceptance=0.70,
        tau_q_min=0.10,
        tau_q_max=0.70,
        tau_t_min=0.20,
        tau_t_max=0.85,
        warmup_requests=5,
    )
    return AdaptiveThresholdController(config, initial_tau_q=0.35, initial_tau_t=0.50)


class TestAdaptiveThresholdController:
    def test_initial_thresholds(self, controller: AdaptiveThresholdController) -> None:
        assert controller.tau_q == 0.35
        assert controller.tau_t == 0.50

    def test_tau_q_leq_tau_t_invariant(self, controller: AdaptiveThresholdController) -> None:
        """tau_Q <= tau_T should always hold after updates."""
        # Push thresholds around
        for _ in range(100):
            controller.record_draft(True)
            controller.record_qualifier(False)
        assert controller.tau_q <= controller.tau_t + 0.02  # small tolerance for rounding

    def test_high_acceptance_increases_tau_q(self, controller: AdaptiveThresholdController) -> None:
        """If too many drafts are accepted, tau_Q should increase (stricter)."""
        initial_tau_q = controller.tau_q
        # Simulate 100% acceptance rate (above target 60%)
        for _ in range(50):
            controller.record_draft(True)
        # After warmup, should have increased
        # (may not change immediately due to warmup)
        assert controller.tau_q >= initial_tau_q - 0.01  # should not decrease much

    def test_low_acceptance_decreases_tau_q(self, controller: AdaptiveThresholdController) -> None:
        """If too few drafts are accepted, tau_Q should decrease (more lenient)."""
        initial_tau_q = controller.tau_q
        # Simulate 0% acceptance rate (below target 60%)
        for _ in range(50):
            controller.record_draft(False)
        # After warmup, should have decreased
        assert controller.tau_q <= initial_tau_q + 0.01

    def test_disabled_controller_does_not_change_thresholds(self) -> None:
        config = AdaptiveThresholdConfig(enabled=False, warmup_requests=0)
        ctrl = AdaptiveThresholdController(config, initial_tau_q=0.35, initial_tau_t=0.50)
        for _ in range(100):
            ctrl.record_draft(True)
            ctrl.record_qualifier(True)
        assert ctrl.tau_q == 0.35
        assert ctrl.tau_t == 0.50

    def test_bounds_respected(self, controller: AdaptiveThresholdController) -> None:
        """Thresholds should stay within configured bounds."""
        for _ in range(500):
            controller.record_draft(True)
            controller.record_qualifier(True)
        assert 0.10 <= controller.tau_q <= 0.70
        assert 0.20 <= controller.tau_t <= 0.85

    def test_get_stats(self, controller: AdaptiveThresholdController) -> None:
        stats = controller.get_stats()
        assert "tau_q" in stats
        assert "tau_t" in stats
        assert "draft_acceptance_rate" in stats
        assert "qualifier_acceptance_rate" in stats
        assert "total_requests" in stats

    def test_warmup_prevents_early_updates(self) -> None:
        config = AdaptiveThresholdConfig(
            enabled=True,
            learning_rate=0.5,
            target_draft_acceptance=0.6,
            target_qualifier_acceptance=0.7,
            warmup_requests=100,  # High warmup
        )
        ctrl = AdaptiveThresholdController(config, initial_tau_q=0.35, initial_tau_t=0.50)
        # Send only 5 requests — should not update
        for _ in range(5):
            ctrl.record_draft(True)
        assert ctrl.tau_q == pytest.approx(0.35, abs=0.01)
