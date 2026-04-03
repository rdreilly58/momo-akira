"""FastAPI server exposing an OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from momo_akira.config import Config
from momo_akira.decoding.engine import PyramidSDEngine


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "momo-akira"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CascadeInfo(BaseModel):
    tokens_per_second: float
    draft_acceptance_rate: float
    qualifier_acceptance_rate: float
    stage1_iterations: int
    outer_iterations: int
    adaptive_tau_q: float
    adaptive_tau_t: float


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    cascade_info: CascadeInfo


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="momo-akira", version="2.0.0")

    # Engine is loaded once at startup
    engine: PyramidSDEngine | None = None

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal engine
        engine = PyramidSDEngine.from_config(cfg)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0"}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")

        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming not yet supported")

        # Build prompt from messages (simple concatenation; swap for proper templates)
        prompt = _messages_to_prompt(req.messages)

        # Override config params if caller specified them
        if req.temperature is not None:
            cfg.decoding.temperature = req.temperature
        if req.max_tokens is not None:
            cfg.decoding.max_new_tokens = req.max_tokens

        result = engine.generate(prompt)
        m = result.metrics
        a = result.adaptive_stats

        # Count prompt tokens
        prompt_tokens = len(engine.tokenizer.encode(prompt))
        completion_tokens = len(result.token_ids)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cascade_info=CascadeInfo(
                tokens_per_second=m.tokens_per_second,
                draft_acceptance_rate=m.draft_acceptance_rate,
                qualifier_acceptance_rate=m.qualifier_acceptance_rate,
                stage1_iterations=m.stage1_iterations,
                outer_iterations=m.outer_iterations,
                adaptive_tau_q=float(a.get("tau_q", cfg.decoding.tau_q)),
                adaptive_tau_t=float(a.get("tau_t", cfg.decoding.tau_t)),
            ),
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert a chat message list to a plain prompt string.

    For Qwen/Llama instruct models the correct approach is to use the
    tokenizer's apply_chat_template().  The engine calls tokenizer.encode()
    on this string directly, so here we produce a reasonable plain-text
    representation that most instruct models handle well.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)
