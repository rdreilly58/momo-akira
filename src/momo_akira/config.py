"""Configuration loading: YAML config file + environment variable overrides.

The YAML config file is the single source of truth for model selection.
Environment variables can override any value using the pattern:
  OPENCLAW_<SECTION>_<KEY>=value
e.g., OPENCLAW_CASCADE_VARIANT=psdf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Model / Provider config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    model_id: str
    provider: Literal["anthropic", "openai", "openrouter"]
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    expected_latency_ms: int = 2000
    timeout_s: int = 30


class ModelsConfig(BaseModel):
    draft: ModelConfig
    qualifier: ModelConfig
    target: ModelConfig


# ---------------------------------------------------------------------------
# Cascade config
# ---------------------------------------------------------------------------


class CascadeConfig(BaseModel):
    variant: Literal["psda", "psdf"] = "psda"
    tau_q: float = Field(0.35, ge=0.0, le=1.0)
    tau_t: float = Field(0.50, ge=0.0, le=1.0)
    speculative_length_d: int = Field(4, ge=1)
    speculative_length_q: int = Field(8, ge=1)

    @model_validator(mode="after")
    def tau_ordering(self) -> "CascadeConfig":
        # Paper recommends tau_Q <= tau_T; warn but don't hard-fail
        if self.tau_q > self.tau_t:
            import warnings

            warnings.warn(
                f"tau_q ({self.tau_q}) > tau_t ({self.tau_t}). "
                "Paper recommends tau_Q <= tau_T for best performance.",
                UserWarning,
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# Confidence scoring config
# ---------------------------------------------------------------------------


class ConfidenceConfig(BaseModel):
    prompt_complexity_weight: float = Field(0.30, ge=0.0, le=1.0)
    response_coherence_weight: float = Field(0.40, ge=0.0, le=1.0)
    logprob_weight: float = Field(0.30, ge=0.0, le=1.0)
    use_logprobs: bool = True
    use_self_scoring: bool = False


# ---------------------------------------------------------------------------
# Adaptive threshold controller config
# ---------------------------------------------------------------------------


class AdaptiveThresholdConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = Field(0.05, gt=0.0, le=1.0)
    target_draft_acceptance: float = Field(0.60, ge=0.0, le=1.0)
    target_qualifier_acceptance: float = Field(0.70, ge=0.0, le=1.0)
    tau_q_min: float = Field(0.10, ge=0.0, le=1.0)
    tau_q_max: float = Field(0.70, ge=0.0, le=1.0)
    tau_t_min: float = Field(0.20, ge=0.0, le=1.0)
    tau_t_max: float = Field(0.85, ge=0.0, le=1.0)
    warmup_requests: int = Field(50, ge=0)


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    log_dir: str = "./logs"
    enabled: bool = True
    include_content: bool = True
    max_bytes: int = 104_857_600  # 100 MB
    backup_count: int = 7


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7780
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    request_timeout_s: int = 120


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    api_key_env: str
    base_url: str | None = None


class ProvidersConfig(BaseModel):
    anthropic: ProviderConfig = ProviderConfig(api_key_env="ANTHROPIC_API_KEY")
    openai: ProviderConfig = ProviderConfig(api_key_env="OPENAI_API_KEY")
    openrouter: ProviderConfig = ProviderConfig(
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    adaptive_threshold: AdaptiveThresholdConfig = Field(
        default_factory=AdaptiveThresholdConfig
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)

    def get_api_key(self, provider: str) -> str | None:
        """Resolve API key for a provider from environment."""
        provider_cfg = getattr(self.providers, provider, None)
        if provider_cfg is None:
            return None
        env_var = provider_cfg.api_key_env
        return os.environ.get(env_var)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply OPENCLAW_<SECTION>_<KEY>=value env var overrides to raw config dict."""
    prefix = "OPENCLAW_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        # Initialize section if absent so env overrides work on minimal configs
        if section not in data:
            data[section] = {}
        if isinstance(data[section], dict):
            # Try to coerce type based on existing value
            existing = data[section].get(field)
            if isinstance(existing, bool):
                data[section][field] = value.lower() in ("1", "true", "yes")
            elif isinstance(existing, int):
                try:
                    data[section][field] = int(value)
                except ValueError:
                    pass
            elif isinstance(existing, float):
                try:
                    data[section][field] = float(value)
                except ValueError:
                    pass
            else:
                data[section][field] = value
    return data


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file with environment variable overrides.

    Search order for config file:
      1. Explicit config_path argument
      2. OPENCLAW_CONFIG env var
      3. ./config.yaml
      4. ~/.openclaw/config.yaml
    """
    if config_path is None:
        config_path = os.environ.get("OPENCLAW_CONFIG")
    if config_path is None and Path("config.yaml").exists():
        config_path = Path("config.yaml")
    if config_path is None:
        fallback = Path.home() / ".openclaw" / "config.yaml"
        if fallback.exists():
            config_path = fallback

    if config_path is None:
        raise FileNotFoundError(
            "No config.yaml found. Create one or set OPENCLAW_CONFIG env var."
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    raw = _apply_env_overrides(raw)
    return AppConfig.model_validate(raw)
