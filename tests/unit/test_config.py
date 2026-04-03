"""Tests for config loading and validation."""

import pytest
from pydantic import ValidationError

from momo_akira.config import Config, DecodingConfig, ModelConfig, ModelsConfig


def _minimal_config() -> dict:
    return {
        "models": {
            "family": "qwen2.5",
            "draft": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
            "qualifier": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
            "target": {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
        }
    }


def test_config_minimal_valid():
    cfg = Config(**_minimal_config())
    assert cfg.models.draft.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.decoding.variant == "psda"
    assert cfg.server.port == 7780


def test_config_defaults():
    cfg = Config(**_minimal_config())
    assert cfg.decoding.speculative_length_d == 4
    assert cfg.decoding.speculative_length_q == 10
    assert cfg.decoding.tau_q == 0.3
    assert cfg.decoding.tau_t == 0.4
    assert cfg.adaptive.enabled is True


def test_decoding_invalid_variant():
    with pytest.raises(ValidationError):
        DecodingConfig(variant="cascade")


def test_decoding_variant_psdf():
    d = DecodingConfig(variant="psdf")
    assert d.variant == "psdf"


def test_model_config_defaults():
    m = ModelConfig(model_id="some/model")
    assert m.device == "auto"
    assert m.dtype == "float16"


def test_config_from_yaml(tmp_path):
    import yaml

    data = _minimal_config()
    data["decoding"] = {"variant": "psdf", "tau_q": 0.2, "tau_t": 0.5}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))

    cfg = Config.from_yaml(p)
    assert cfg.decoding.variant == "psdf"
    assert cfg.decoding.tau_q == pytest.approx(0.2)
    assert cfg.decoding.tau_t == pytest.approx(0.5)
