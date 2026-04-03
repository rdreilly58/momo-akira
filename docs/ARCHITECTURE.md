# Architecture: Momo-Akira

## Overview

Momo-Akira is a production-ready OpenAI-compatible proxy that implements PyramidSD-style 3-tier speculative decoding for API-based LLM inference. It sits transparently between your application and upstream LLM providers, routing each request through the cheapest capable model tier while preserving response quality.

The system is adapted from the PyramidSD paper (arxiv:2510.12966, NeurIPS 2025). The key insight is that a large fraction of real-world queries are simple enough to be answered confidently by smaller, faster models. By measuring response confidence at each tier and only escalating when confidence is insufficient, the proxy dramatically reduces cost and latency on the majority of requests without degrading quality on complex ones.

---

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client Application                             │
│              (any OpenAI-compatible SDK or HTTP client)                 │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │  POST /v1/chat/completions
                          │  (standard OpenAI request format)
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  OpenClaw Proxy  (localhost:7780)                       │
│                                                                         │
│  ┌─────────────┐   ┌──────────────────────────────────────────────┐    │
│  │  FastAPI     │   │              Cascade Pipeline                │    │
│  │  Server      │──▶│  ┌──────────┐  ┌───────────┐  ┌─────────┐  │    │
│  │  (server.py) │   │  │  Draft   │  │ Qualifier │  │ Target  │  │    │
│  └─────────────┘   │  │  Tier    │─▶│  Tier     │─▶│  Tier   │  │    │
│                    │  │ (fast)   │  │ (medium)  │  │(accurate│  │    │
│  ┌─────────────┐   │  └──────────┘  └───────────┘  └─────────┘  │    │
│  │  Config     │   │        │              │              │       │    │
│  │  (YAML +    │   │  Confidence    Confidence      Final resp.   │    │
│  │  env vars)  │   │  scoring       scoring                       │    │
│  └─────────────┘   └──────────────────────────────────────────────┘    │
│                                    │                                    │
│  ┌─────────────┐   ┌───────────────▼──────────────────────────────┐    │
│  │  Metrics    │   │            Provider Layer                    │    │
│  │  & Logging  │   │  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │    │
│  │  (JSONL)    │   │  │Anthropic │ │  OpenAI  │ │ OpenRouter  │  │    │
│  └─────────────┘   │  └──────────┘ └──────────┘ └─────────────┘  │    │
│                    └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                          │  Response with cascade_info metadata
                          ▼
                    Client Application
```

---

## 3-Tier Cascade Diagram

Each request flows through the cascade from left to right. At each tier, the confidence score C is computed and compared to the tier's threshold. If C meets the threshold, the response is returned immediately without escalating.

```
Request
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1: DRAFT  (e.g., claude-haiku-3-5, gpt-4o-mini)                  │
│                                                                         │
│  1. Send request to draft model                                         │
│  2. Compute confidence score C_D from:                                  │
│     - Prompt complexity analysis                                        │
│     - Response coherence scoring                                        │
│     - Logprobs (if available from provider)                             │
│     - Self-consistency check                                            │
│                                                                         │
│  C_D >= tau_Q ?                                                         │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
         YES ──▶│ Return draft response (fast path, ~60-80% of requests)
                │
         NO ────▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 2: QUALIFIER  (e.g., claude-sonnet-3-5, gpt-4o)                  │
│                                                                         │
│  PSDA: Generate fresh qualifier response                                │
│  PSDF: Draft response may still be returned if "fuzzy" check passes     │
│                                                                         │
│  1. Send request to qualifier model (PSDA) or evaluate draft (PSDF)    │
│  2. Compute confidence score C_Q                                        │
│                                                                         │
│  C_Q >= tau_T ?                                                         │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
         YES ──▶│ Return qualifier response (~15-30% of requests)
                │
         NO ────▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 3: TARGET  (e.g., claude-opus-4, gpt-4-turbo)                    │
│                                                                         │
│  1. Send request to target model (full quality, no gating)              │
│  2. Return response unconditionally (~5-15% of requests)                │
└─────────────────────────────────────────────────────────────────────────┘
```

Thresholds satisfy `tau_Q <= tau_T` (from paper Section 4), meaning the qualifier gate is strictly easier to pass than the target gate.

---

## Component Diagram

```
src/momo_akira/
│
├── server.py
│   └── FastAPI application, route definitions, request/response shaping
│       Injects cascade_info metadata into responses
│
├── config.py
│   └── Loads config.yaml and environment variables
│       Provides a single validated Config dataclass to all components
│
├── metrics.py
│   └── Per-request cost tracking, daily cost accumulation,
│       acceptance rate histograms by tier
│
├── models/
│   ├── base.py         Abstract LLMProvider interface (complete, chat, etc.)
│   ├── anthropic.py    Anthropic Messages API adapter
│   ├── openai.py       OpenAI Chat Completions adapter
│   └── openrouter.py   OpenRouter adapter (wraps OpenAI-compat endpoint)
│
├── cascade/
│   ├── pipeline.py     CascadePipeline: orchestrates tier calls and gating
│   ├── confidence.py   ConfidenceScorer: composite C ∈ [0,1] computation
│   ├── divergence.py   DivergenceMeasures: TV distance and KL-div proxies
│   ├── threshold.py    AdaptiveThresholdController: EMA-based tau tuning
│   └── variants.py     PsdaVariant and PsdfVariant strategy classes
│
└── logging_/
    └── logger.py       JSONL structured logger with daily rotation
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `server.py` | HTTP interface, OpenAI compatibility shim, streaming support |
| `config.py` | Single source of truth for all runtime configuration |
| `pipeline.py` | Cascade state machine, tier sequencing, graceful degradation |
| `confidence.py` | Computes the proxy for Div(P_i, P_j) without raw logits |
| `divergence.py` | Divergence measure implementations used by confidence scorer |
| `threshold.py` | Online learning of tau_Q and tau_T from acceptance rates |
| `variants.py` | PSDA and PSDF branch logic as separate strategy objects |
| `base.py` | Provider interface contract (async complete, stream) |
| `logger.py` | Structured request/response logging for observability |
| `metrics.py` | Cost accounting and performance counters |

---

## Request Lifecycle

The following describes the complete lifecycle of a `POST /v1/chat/completions` request.

```
1. HTTP Receive
   - FastAPI receives request on /v1/chat/completions
   - Request is parsed into an internal ChatRequest dataclass
   - A request_id (UUID) is assigned

2. Config Resolution
   - Active cascade variant (PSDA or PSDF) is read from config
   - Current tau_Q and tau_T are fetched from AdaptiveThresholdController

3. Draft Tier Invocation
   - The draft model provider is called asynchronously
   - Response text is captured (and logprobs if supported)
   - ConfidenceScorer.score(request, draft_response) → C_D

4. Draft Gate
   - If C_D >= tau_Q: record tier=draft, return immediately (fast path)
   - Otherwise: continue to qualifier tier

5. Qualifier Tier Invocation  (PSDA path)
   - A fresh request is sent to the qualifier model
   - ConfidenceScorer.score(request, qualifier_response) → C_Q

   (PSDF path)
   - Draft response is evaluated for fuzzy acceptance
   - If fuzzy check passes, draft response may be returned here

6. Qualifier Gate
   - If C_Q >= tau_T: record tier=qualifier, return qualifier response
   - Otherwise: continue to target tier

7. Target Tier Invocation
   - Request is sent to target model unconditionally
   - Response is returned directly

8. Response Shaping
   - The winning response is wrapped in OpenAI ChatCompletion format
   - cascade_info metadata is injected into the response object:
       { "tier": "draft"|"qualifier"|"target",
         "confidence": float,
         "draft_ms": int,
         "qualifier_ms": int|null,
         "target_ms": int|null,
         "cost_usd": float }

9. Logging
   - Full JSONL record written to log file (request, response, cascade_info)

10. Metrics Update
    - Cost counters incremented
    - Acceptance rate EMA updated in AdaptiveThresholdController
    - Tier histogram counter incremented
```

---

## Data Flow: PSDA Variant

PSDA (PyramidSD Assisted) maintains a quality floor equal to the qualifier model. When the draft is rejected, the qualifier generates a completely independent response.

```
Request ──────────────────────────────────────────────────────────────────▶

         ┌─── Draft Call ───┐
         │  model: haiku    │
         │  C_D computed    │
         └──────────────────┘
                │
         C_D >= tau_Q?
                │
         YES ───┴──────────────────────────────────────── Return draft
                │
         NO     │
                ▼
         ┌─── Qualifier Call ───┐    (fresh generation, not draft-assisted)
         │  model: sonnet        │
         │  C_Q computed         │
         └───────────────────────┘
                │
         C_Q >= tau_T?
                │
         YES ───┴──────────────────────────────────────── Return qualifier
                │
         NO     │
                ▼
         ┌─── Target Call ───┐
         │  model: opus       │
         │  unconditional     │
         └────────────────────┘
                │
                └─────────────────────────────────────── Return target
```

## Data Flow: PSDF Variant

PSDF (PyramidSD Fuzzy) is more aggressive. Draft responses that pass a relaxed fuzzy check are returned at the qualifier stage without generating a new qualifier response. This is faster but can allow lower-quality draft responses through.

```
Request ──────────────────────────────────────────────────────────────────▶

         ┌─── Draft Call ───┐
         │  model: haiku    │
         │  C_D computed    │
         └──────────────────┘
                │
         C_D >= tau_Q?
                │
         YES ───┴──────────────────────────────────────── Return draft
                │
         NO     │
                ▼
         ┌─── Fuzzy Evaluation ─────────────────────────────────────┐
         │  Re-score draft response against qualifier-level criteria │
         │  C_fuzzy = divergence_proxy(draft, qualifier_context)     │
         └──────────────────────────────────────────────────────────┘
                │
         C_fuzzy >= tau_T?  (draft survives fuzzy gate)
                │
         YES ───┴──────────────────────────────────────── Return draft
                │                                          (fuzzy-accepted)
         NO     │
                ▼
         ┌─── Qualifier Call ───┐
         │  model: sonnet        │
         │  C_Q computed         │
         └───────────────────────┘
                │
         C_Q >= tau_T?
                │
         YES ───┴──────────────────────────────────────── Return qualifier
                │
         NO     │
                ▼
         ┌─── Target Call ───┐   (unconditional)
         └────────────────────┘
                └─────────────────────────────────────── Return target
```

---

## Provider Abstraction Layer

All LLM calls are routed through a common `LLMProvider` abstract base class defined in `models/base.py`. This decouples the cascade logic from provider-specific API details.

```
LLMProvider (base.py)
├── async complete(request) → LLMResponse
├── async stream(request) → AsyncIterator[LLMChunk]
└── supports_logprobs() → bool

        ┌──────────────────────────────────────────┐
        │                                          │
AnthropicProvider    OpenAIProvider    OpenRouterProvider
(Messages API)       (Chat Completions) (OpenAI-compat,
                                         multi-model)
```

Each provider adapter handles:
- Authentication (API key injection from environment)
- Request format translation (internal → provider-specific)
- Response normalization (provider-specific → internal `LLMResponse`)
- Logprobs extraction when available (OpenAI supports `logprobs=true`)
- Error mapping to internal exception types for graceful degradation

The pipeline selects providers at startup based on `config.yaml` model assignments. Draft, qualifier, and target models can each use different providers.

---

## Configuration System

Configuration is loaded once at startup by `config.py` and exposed as an immutable `Config` dataclass. The loading order is:

```
1. config.yaml (base values, checked into version control with safe defaults)
2. Environment variables (override config.yaml for secrets and deployment tuning)
3. Runtime overrides (AdaptiveThresholdController may update tau_Q/tau_T in memory)
```

The `Config` object is dependency-injected into all components via FastAPI's dependency system, making it straightforward to test with alternate configurations.

```
config.yaml
    │
    ├── models:     draft, qualifier, target model IDs and providers
    ├── cascade:    tau_Q, tau_T, variant, timeouts, speculative_lengths
    ├── confidence: signal weights (complexity, coherence, logprobs, self_score)
    ├── adaptive_threshold: enabled, target_acceptance_rate, ema_alpha
    ├── server:     host, port, log_level, workers
    ├── logging:    path, rotation, format
    └── providers:  API key env var names (never literal key values)
```

---

## Metrics and Logging Pipeline

Every request produces two side-effect outputs: a structured log record and metric updates.

```
Request completes
        │
        ├──▶ JSONL Logger (logging_/logger.py)
        │        │
        │        └── Appends one JSON line to daily log file:
        │            {timestamp, request_id, tier, confidence,
        │             prompt_tokens, completion_tokens, cost_usd,
        │             latency_ms, model_ids, cascade_info}
        │
        └──▶ Metrics (metrics.py)
                 │
                 ├── Increment tier counter (draft/qualifier/target)
                 ├── Accumulate cost_usd (per-request + daily total)
                 ├── Update latency histogram bucket
                 └── Feed acceptance rate to AdaptiveThresholdController
                          │
                          └── EMA update of tau_Q and tau_T
                              (if adaptive_threshold.enabled = true)
```

Metrics are exposed at `GET /v1/metrics` as a JSON snapshot. Log files are rotated daily with the pattern `openclaw-YYYY-MM-DD.jsonl`.
