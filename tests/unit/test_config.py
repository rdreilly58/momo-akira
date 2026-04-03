"""Tests for configuration loading."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from momo_akira.config import AppConfig, CascadeConfig, load_config


def write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


def minimal_config_data() -> dict:
    return {
        "models": {
            "draft": {
                "model_id": "haiku",
                "provider": "anthropic",
            },
            "qualifier": {
                "model_id": "sonnet",
                "provider": "anthropic",
            },
            "target": {
                "model_id": "opus",
                "provider": "anthropic",
            },
        }
    }


def test_load_minimal_config(tmp_path: Path) -> None:
    p = write_config(tmp_path, minimal_config_data())
    config = load_config(p)
    assert config.models.draft.model_id == "haiku"
    assert config.models.qualifier.model_id == "sonnet"
    assert config.models.target.model_id == "opus"


def test_defaults_applied(tmp_path: Path) -> None:
    p = write_config(tmp_path, minimal_config_data())
    config = load_config(p)
    assert config.cascade.variant == "psda"
    assert config.cascade.tau_q == 0.35
    assert config.cascade.tau_t == 0.50
    assert config.server.port == 7780


def test_tau_ordering_warning(tmp_path: Path) -> None:
    data = minimal_config_data()
    data["cascade"] = {"tau_q": 0.6, "tau_t": 0.3}
    p = write_config(tmp_path, data)
    with pytest.warns(UserWarning, match="tau_Q.*tau_T"):
        config = load_config(p)
    assert config.cascade.tau_q == 0.6
    assert config.cascade.tau_t == 0.3


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_config(tmp_path, minimal_config_data())
    monkeypatch.setenv("OPENCLAW_CASCADE_VARIANT", "psdf")
    monkeypatch.setenv("OPENCLAW_SERVER_PORT", "9999")
    config = load_config(p)
    assert config.cascade.variant == "psdf"
    assert config.server.port == 9999


def test_provider_swap(tmp_path: Path) -> None:
    """Swapping models only requires changing config — no code changes."""
    data = minimal_config_data()
    data["models"]["draft"] = {
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    }
    p = write_config(tmp_path, data)
    config = load_config(p)
    assert config.models.draft.model_id == "gpt-4o-mini"
    assert config.models.draft.provider == "openai"
    # Qualifier and target unchanged
    assert config.models.qualifier.provider == "anthropic"


def test_get_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_config(tmp_path, minimal_config_data())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    config = load_config(p)
    assert config.get_api_key("anthropic") == "test-key-123"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_invalid_provider(tmp_path: Path) -> None:
    data = minimal_config_data()
    data["models"]["draft"]["provider"] = "unknown_provider"
    p = write_config(tmp_path, data)
    with pytest.raises(Exception):
        load_config(p)
