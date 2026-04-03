# openclaw-speculative-decoder

An OpenAI-compatible proxy that implements **PyramidSD-style 3-tier speculative decoding** for API-based LLMs. Inspired by the paper [PyramidSD (arxiv:2510.12966, NeurIPS 2025)](https://arxiv.org/abs/2510.12966).

Instead of sending every request to an expensive model, the proxy routes through a cascade:

```
Request → Draft (Haiku) → Qualifier (Sonnet) → Target (Opus)
                ↓                   ↓                ↓
           Accept if              Accept if       Always
           confident              confident       accept
```

Cheap requests resolve at the draft tier. Hard requests escalate only as far as needed. You get target-quality answers where it matters, at a fraction of the cost.

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY (and/or OPENAI_API_KEY, OPENROUTER_API_KEY)

# 3. Start the proxy
openclaw-decoder --config config.yaml
# Listening on http://127.0.0.1:7780

# 4. Use it exactly like the OpenAI API
curl http://127.0.0.1:7780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cascade",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

## Response example

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "cascade/draft",
  "choices": [{
    "message": {"role": "assistant", "content": "Paris is the capital of France."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
  "cascade_info": {
    "accepted_tier": "draft",
    "variant": "psda",
    "tiers_invoked": {"draft": true, "qualifier": false, "target": false},
    "confidence": {"draft": 0.82, "qualifier": null},
    "cost_usd": 0.000009,
    "latency_ms": 412.3
  }
}
```

## Swapping models

Open `config.yaml` and change the model — no code changes required:

```yaml
models:
  draft:
    model_id: "gpt-4o-mini"    # was claude-haiku
    provider: "openai"          # was anthropic
  qualifier:
    model_id: "gpt-4o"
    provider: "openai"
  target:
    model_id: "claude-opus-4-6"
    provider: "anthropic"
```

Restart and it works.

## Configuration

Key settings in `config.yaml`:

```yaml
cascade:
  variant: "psda"   # psda (stable) or psdf (faster, more variance)
  tau_q: 0.35       # draft acceptance threshold — lower = more drafts accepted
  tau_t: 0.50       # qualifier acceptance threshold
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference.

## Algorithm

Adapted from PyramidSD (paper Section 3):

1. **Draft** generates a response. A composite confidence score `C ∈ [0,1]` is computed (prompt complexity + response coherence + optional logprobs).
2. If `C >= tau_Q` → accept the draft response (done, cheap).
3. **Qualifier** generates a response (PSDA) or evaluates the draft (PSDF).
4. If `C >= tau_T` → accept the qualifier response (done, moderate cost).
5. **Target** generates the final response (expensive, highest quality).

The paper's key insight: `tau_Q <= tau_T` works best — be strict at the first gate, more lenient at the final gate.

See [docs/CASCADE_ALGORITHM.md](docs/CASCADE_ALGORITHM.md) for full details.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Main inference (OpenAI compatible) |
| `GET` | `/health` | Health check + current config |
| `GET` | `/v1/metrics` | Cost, acceptance rates, threshold stats |
| `GET` | `/v1/models` | Available models |

## Development

```bash
make dev        # Install with dev dependencies + pre-commit hooks
make test       # Run full test suite
make test-unit  # Unit tests only
make lint       # Ruff lint
make typecheck  # mypy
make fmt        # Auto-format
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Design Decisions](docs/DESIGN_DECISIONS.md)
- [Cascade Algorithm](docs/CASCADE_ALGORITHM.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [API Reference](docs/API_REFERENCE.md)
