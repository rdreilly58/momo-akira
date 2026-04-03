"""FastAPI application: OpenAI-compatible proxy with PyramidSD cascade.

Endpoints:
  POST /v1/chat/completions  — Main inference endpoint (OpenAI format)
  GET  /health               — Health check
  GET  /v1/metrics           — Metrics and cost data
  GET  /v1/models            — Available models (cascade config)

The response includes a cascade_info field in the response body with
metadata about which tier was used, confidence scores, and cost.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .cascade.pipeline import CascadePipeline
from .config import AppConfig, load_config
from .logging_.logger import RequestLogger
from .metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str = "cascade"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    # OpenAI-compatible extras (passed through to providers)
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CascadeInfo(BaseModel):
    accepted_tier: str
    variant: str
    tiers_invoked: dict[str, bool]
    confidence: dict[str, float | None]
    cost_usd: float
    latency_ms: float


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    # Extension: cascade metadata
    cascade_info: CascadeInfo | None = None


# ---------------------------------------------------------------------------
# App state (injected via lifespan)
# ---------------------------------------------------------------------------


class AppState:
    def __init__(self) -> None:
        self.config: AppConfig | None = None
        self.pipeline: CascadePipeline | None = None
        self.logger: RequestLogger | None = None
        self.metrics: MetricsCollector | None = None


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and initialize cascade pipeline on startup."""
    config = load_config()
    _state.config = config
    _state.pipeline = CascadePipeline(config)
    _state.logger = RequestLogger(config.logging)
    _state.metrics = MetricsCollector()
    yield
    # Cleanup on shutdown (nothing needed currently)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="openclaw-speculative-decoder",
    description="OpenAI-compatible proxy with PyramidSD 3-tier speculative decoding",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    config = _state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {
        "status": "ok",
        "version": "0.1.0",
        "cascade_variant": config.cascade.variant,
        "tiers": {
            "draft": config.models.draft.model_id,
            "qualifier": config.models.qualifier.model_id,
            "target": config.models.target.model_id,
        },
        "thresholds": {
            "tau_q": config.cascade.tau_q,
            "tau_t": config.cascade.tau_t,
        },
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    config = _state.config
    if config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    models = []
    for tier_name in ("draft", "qualifier", "target"):
        tier_cfg = getattr(config.models, tier_name)
        models.append({
            "id": f"cascade-{tier_name}",
            "object": "model",
            "created": 0,
            "owned_by": tier_cfg.provider,
            "tier": tier_name,
            "model_id": tier_cfg.model_id,
        })
    # Also expose the cascade model
    models.insert(0, {
        "id": "cascade",
        "object": "model",
        "created": 0,
        "owned_by": "openclaw",
        "description": "3-tier PyramidSD cascade",
    })
    return {"object": "list", "data": models}


@app.get("/v1/metrics")
async def get_metrics() -> dict[str, Any]:
    if _state.metrics is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    threshold_stats = _state.pipeline.get_threshold_stats() if _state.pipeline else {}
    return {
        "metrics": _state.metrics.get_summary(),
        "adaptive_thresholds": threshold_stats,
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
    if _state.pipeline is None or _state.config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not yet supported")

    # Convert messages to dict format
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    # Separate system message if present
    system: str | None = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = str(msg["content"])
        else:
            filtered_messages.append(msg)

    # Build extra params (forward OpenAI-compatible fields)
    extra_params: dict[str, Any] = {}
    if request.top_p is not None:
        extra_params["top_p"] = request.top_p
    if request.stop is not None:
        extra_params["stop"] = request.stop

    request_id = RequestLogger.new_request_id()

    # Run cascade
    result = await _state.pipeline.run(
        messages=filtered_messages if filtered_messages else messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system=system,
        extra_params=extra_params,
    )

    # Record metrics and log
    _state.metrics.record(result)
    if _state.logger:
        _state.logger.log_request(
            request_id=request_id,
            messages=messages,
            result=result,
        )

    # Build OpenAI-format response
    response_body = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"cascade/{result.accepted_tier.value}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response.content,
                },
                "finish_reason": "stop" if result.response.success else "error",
            }
        ],
        "usage": {
            "prompt_tokens": result.response.input_tokens,
            "completion_tokens": result.response.output_tokens,
            "total_tokens": result.response.total_tokens,
        },
        "cascade_info": result.to_metadata(),
    }

    status_code = 200 if result.response.success else 500
    return JSONResponse(content=response_body, status_code=status_code)
