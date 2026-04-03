"""Tests for the adaptive threshold controller."""

import pytest

from momo_akira.config import AdaptiveConfig, DecodingConfig
from momo_akira.decoding.adaptive import AdaptiveThresholdController


def _make_controller(
    tau_q: float = 0.3,
    tau_t: float = 0.4,
    enabled: bool = True,
    warmup: int = 0,
    lr: float = 0.1,
    target_q: float = 0.70,
    target_t: float = 0.80,
) -> AdaptiveThresholdController:
    dec = DecodingConfig(tau_q=tau_q, tau_t=tau_t)
    ada = AdaptiveConfig(
        enabled=enabled,
        learning_rate=lr,
        warmup_tokens=warmup,
        target_draft_acceptance=target_q,
        target_qualifier_acceptance=target_t,
    )
    return AdaptiveThresholdController(dec, ada)


def test_initial_thresholds():
    ctrl = _make_controller(tau_q=0.3, tau_t=0.4)
    assert ctrl.tau_q == pytest.approx(0.3)
    assert ctrl.tau_t == pytest.approx(0.4)


def test_disabled_no_change():
    ctrl = _make_controller(enabled=False, warmup=0)
    ctrl.update(draft_accepted=5, draft_proposed=10, qualifier_accepted=5, qualifier_proposed=10)
    assert ctrl.tau_q == pytest.approx(0.3)
    assert ctrl.tau_t == pytest.approx(0.4)


def test_warmup_prevents_update():
    ctrl = _make_controller(warmup=1000, lr=0.5)
    ctrl.update(draft_accepted=1, draft_proposed=10, qualifier_accepted=1, qualifier_proposed=10)
    # Still in warmup — thresholds should not move
    assert ctrl.tau_q == pytest.approx(0.3)


def test_low_acceptance_lowers_tau():
    """Below-target acceptance should lower tau (be more permissive)."""
    ctrl = _make_controller(tau_q=0.3, target_q=0.70, warmup=0, lr=0.1)
    # 20% acceptance rate (well below 70% target)
    ctrl.update(draft_accepted=2, draft_proposed=10, qualifier_accepted=8, qualifier_proposed=10)
    assert ctrl.tau_q < 0.3


def test_high_acceptance_raises_tau():
    """Above-target acceptance should raise tau (be stricter)."""
    ctrl = _make_controller(tau_q=0.3, target_q=0.70, warmup=0, lr=0.1)
    # 100% acceptance rate (above 70% target)
    ctrl.update(draft_accepted=10, draft_proposed=10, qualifier_accepted=8, qualifier_proposed=10)
    assert ctrl.tau_q > 0.3


def test_stats_contains_keys():
    ctrl = _make_controller()
    s = ctrl.stats
    assert "tau_q" in s
    assert "tau_t" in s
    assert "ema_acceptance_q" in s
    assert "ema_acceptance_t" in s
