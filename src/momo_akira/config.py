"""Configuration models for momo-akira v2."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    model_id: str
    device: str = "auto"
    dtype: str = "float16"


class ModelsConfig(BaseModel):
    family: str
    draft: ModelConfig
    qualifier: ModelConfig
    target: ModelConfig


class DecodingConfig(BaseModel):
    variant: str = "psda"  # "psda" | "psdf"
    speculative_length_d: int = 4   # l_D: tokens drafted per inner batch
    speculative_length_q: int = 10  # l_Q: qualifier-verified tokens before target
    tau_q: float = 0.3              # draft→qualifier divergence threshold
    tau_t: float = 0.4              # qualifier→target divergence threshold
    temperature: float = 0.7
    max_new_tokens: int = 2048

    @field_validator("variant")
    @classmethod
    def variant_must_be_valid(cls, v: str) -> str:
        if v not in ("psda", "psdf"):
            raise ValueError("variant must be 'psda' or 'psdf'")
        return v

    @field_validator("tau_t")
    @classmethod
    def tau_ordering(cls, v: float, info: object) -> float:
        # Soft check: paper recommends tau_q <= tau_t
        return v


class AdaptiveConfig(BaseModel):
    enabled: bool = True
    target_draft_acceptance: float = 0.70
    target_qualifier_acceptance: float = 0.80
    learning_rate: float = 0.05
    warmup_tokens: int = 500


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7780


class LoggingConfig(BaseModel):
    log_dir: str = "./logs"
    enabled: bool = True


class Config(BaseModel):
    models: ModelsConfig
    decoding: DecodingConfig = DecodingConfig()  # type: ignore[call-arg]
    adaptive: AdaptiveConfig = AdaptiveConfig()  # type: ignore[call-arg]
    server: ServerConfig = ServerConfig()        # type: ignore[call-arg]
    logging: LoggingConfig = LoggingConfig()     # type: ignore[call-arg]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
