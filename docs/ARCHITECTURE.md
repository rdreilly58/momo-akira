# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                 OpenAI-Compatible API                         │
│             POST /v1/chat/completions                         │
│             (FastAPI — src/momo_akira/server.py)              │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│              PyramidSD Engine                                 │
│          (src/momo_akira/decoding/engine.py)                  │
│                                                              │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────┐        │
│  │  DraftModel │──▶│VerifierModel│──▶│VerifierModel │        │
│  │  (M_D)      │   │   (M_Q)     │   │   (M_T)      │        │
│  │  0.5B       │   │    3B       │   │    7B        │        │
│  └────────────┘   └─────────────┘   └──────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Shared Tokenizer                        │       │
│  │  (loaded from draft model_id, shared across all)  │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  DivergenceChecker (×2)                           │       │
│  │  Div(P_A, P_B) = |logit_A(x_t) - logit_B(x_t)|  │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  KVCacheManager (×3 — one per model)              │       │
│  │  Manages past_key_values, commit/rollback         │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  AdaptiveThresholdController                      │       │
│  │  EMA tracking → adjusts τ_Q, τ_T at runtime      │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## Module Map

```
src/momo_akira/
├── __init__.py           version string
├── cli.py                click CLI → uvicorn launcher
├── config.py             pydantic Config models, from_yaml()
├── server.py             FastAPI app, OpenAI-compatible endpoints
├── models/
│   ├── draft.py          DraftModel — sequential token generation + logits
│   └── verifier.py       VerifierModel — parallel verification forward pass
└── decoding/
    ├── divergence.py     DivergenceChecker — Div() + accept/reject + PSDA/PSDF
    ├── cache.py          KVCacheManager — commit / rollback past_key_values
    ├── adaptive.py       AdaptiveThresholdController — EMA threshold tuning
    └── engine.py         PyramidSDEngine — orchestrates the full loop
```

## Request Lifecycle

1. `POST /v1/chat/completions` received by FastAPI
2. Messages concatenated into prompt string
3. Prompt tokenized via shared tokenizer
4. `engine.generate(prompt)` called
5. Outer loop: Stage 1 (draft→qualifier) then Stage 2 (qualifier→target)
6. Output token IDs decoded back to text
7. Response returned with OpenAI-compatible shape + `cascade_info` metadata

## KV Cache Strategy

Each model maintains an independent `KVCacheManager`.  After each successful
Stage 2 acceptance, the cache is committed up to the new length.  On rejection,
the cache is trimmed back to the last committed position (no wasteful recomputation).

The draft model's cache is managed independently from the qualifier's because
they process different views of the sequence (draft generates; qualifier verifies).
