# momo-akira v2

OpenAI-compatible local inference server implementing PyramidSD — token-level
3-tier speculative decoding (arxiv:2510.12966).

A 7B target model serving tokens at ~2x speed by using a 0.5B draft and a 3B
qualifier to speculate ahead, both verified in parallel.

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Start server (models are downloaded from HuggingFace on first run)
momo-akira --config config.yaml
```

Server starts on `http://127.0.0.1:7780`.

## Usage

```bash
curl http://127.0.0.1:7780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "momo-akira",
    "messages": [{"role": "user", "content": "What is speculative decoding?"}]
  }'
```

Response includes a `cascade_info` field with per-request metrics:

```json
{
  "cascade_info": {
    "tokens_per_second": 47.3,
    "draft_acceptance_rate": 0.71,
    "qualifier_acceptance_rate": 0.83,
    "stage1_iterations": 12,
    "outer_iterations": 4,
    "adaptive_tau_q": 0.31,
    "adaptive_tau_t": 0.40
  }
}
```

## How It Works

1. **Draft (0.5B)** proposes `l_D` tokens sequentially (fast, cheap)
2. **Qualifier (3B)** verifies all `l_D` in one forward pass
   — accepts tokens where `|logit_D - logit_Q| ≤ τ_Q`
3. Once `l_Q` qualifier-verified tokens accumulate, **Target (7B)** verifies
   them all in one forward pass — accepts where `|logit_Q - logit_T| ≤ τ_T`
4. Final accepted tokens are output; KV caches are committed on acceptance,
   rolled back on rejection.

The speedup comes from verifying K tokens in the time it takes to generate 1.

See [docs/ALGORITHM.md](docs/ALGORITHM.md) for the full walkthrough.

## Configuration

Edit `config.yaml`. All three models **must share the same tokenizer family**.

Default: `Qwen/Qwen2.5-0.5B-Instruct` / `3B` / `7B-Instruct`
(auto-downloaded from HuggingFace).

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all options.

## Development

```bash
make dev        # install with dev deps + pre-commit
make test       # run test suite
make lint       # ruff check
make typecheck  # mypy
```

## Requirements

- Python 3.11+
- PyTorch 2.2+ (MPS / CUDA / CPU)
- ~8 GB RAM/VRAM for all three models in float16
- HuggingFace Transformers 4.45+
