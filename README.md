# momo-akira

**Token-level 3-model speculative decoding for Apple Silicon, based on the PyramidSD paper (arxiv:2510.12966).**

## What This Is

An implementation of Pyramid Speculative Decoding — a technique that accelerates LLM inference by using three models (draft → qualifier → target) to verify tokens in parallel. The draft model proposes tokens cheaply, the qualifier filters them, and the target does final verification. Because GPU forward passes process multiple tokens nearly as fast as one, verifying K tokens costs roughly the same as generating 1.

This implementation runs entirely on Apple Silicon (M4 Mac mini, 16GB) using MLX with 4-bit quantized models from the Qwen2.5 family.

## Current Status (April 3, 2026)

### ✅ Working
- **All 3 models load successfully** on M4 Mac mini with 16GB RAM
  - Draft: Qwen2.5-0.5B-Instruct-4bit
  - Qualifier: Qwen2.5-1.5B-Instruct-4bit  
  - Target: Qwen2.5-7B-Instruct-4bit
- **Peak memory: 5.56 GB** — well within 16GB budget
- **Full PyramidSD loop executes** end-to-end (draft→qualifier→target pipeline)
- **31 unit/integration tests passing** (81% coverage) for the core algorithm
- **2-model speculative decoding works** via mlx-lm's built-in `--draft-model` flag

### 🐛 Known Bug: KV Cache Desync
The 3-model `pyramid_mlx.py` script has a cache synchronization bug:
- When draft tokens are rejected by the qualifier, the draft model's KV cache has already advanced past those positions
- The qualifier and target caches haven't consumed the rejected tokens
- This causes the caches to drift out of sync, producing degraded output
- **Fix needed:** Implement proper cache trim/rollback — on rejection at position N, trim all caches back to position N before continuing

### ❌ Not Yet Working
- Cache rollback on token rejection (the critical remaining piece)
- FastAPI server with local models (PyTorch version OOMs; MLX version needs the cache fix)
- Benchmark suite comparing baseline vs 2-model vs 3-model PyramidSD

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          User Prompt                     │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │          PyramidSD Engine                       │
              │                                                │
              │  Outer loop (until max_tokens or EOS):         │
              │                                                │
              │  STAGE 1: Draft → Qualifier (inner loop)       │
              │    1. Draft (0.5B) generates l_D tokens        │
              │       sequentially (l_D fast forward passes)   │
              │    2. Qualifier (1.5B) verifies all l_D in     │
              │       ONE forward pass                         │
              │    3. Per-token: Div(logit_D, logit_Q) ≤ τ_Q? │
              │       Accept or reject + PSDA fallback         │
              │    4. Repeat until l_Q tokens accumulated      │
              │                                                │
              │  STAGE 2: Qualifier buffer → Target            │
              │    1. Target (7B) verifies all l_Q in ONE pass │
              │    2. Compare P_T vs P_Q (NOT P_D!)            │
              │    3. Per-token: Div(logit_Q, logit_T) ≤ τ_T?  │
              │    4. Accepted tokens → final output            │
              └────────────────────────────────────────────────┘
```

## Key Insight from the Paper

In Stage 2, the target compares against the **qualifier's** distributions (P_Q), not the draft's (P_D). Since P_Q is closer to P_T than P_D is, more tokens pass verification. This is the whole reason the qualifier model exists — it bridges the distributional gap.

## Benchmarks (Preliminary, April 3, 2026)

| Mode | Speed | Memory | Notes |
|------|-------|--------|-------|
| 7B baseline (no speculation) | 25.0 tok/s | 4.4 GB | mlx-lm direct generation |
| 2-model (0.5B→7B) | 25.0 tok/s | 4.7 GB | Too large distributional gap |
| 2-model (0.5B→3B) | 45.9-67.6 tok/s | 2.1 GB | Works well, 3B is fast enough that speculation overhead hurts |
| 3-model PyramidSD | 4.8 tok/s | 5.6 GB | **Bug** — cache desync causing massive rejection rates |

The 2-model results confirm the paper's finding: speculative decoding helps most when the target model is slow. The 3B at 55 tok/s is already fast enough on Apple Silicon that speculation doesn't help. The 7B at 25 tok/s is a better target — once the cache bug is fixed, PyramidSD should show meaningful speedup here.

## Lessons Learned

### 1. Speculative Decoding ≠ Model Routing
Our first attempt (v1) built a **response-level cascade router** — generate a full response from a cheap model, score it, escalate if bad. This is fundamentally NOT speculative decoding. The real algorithm operates at the **token level**: draft proposes K tokens, verifier checks them in parallel via logit distribution comparison.

### 2. API Models Can't Do This
Speculative decoding requires direct access to per-token logit distributions and parallel forward passes. Cloud API models (Claude, GPT) don't expose this. This must run on **local models** with a shared tokenizer.

### 3. Memory Management is Critical
On 16GB machines, fitting 3 models requires:
- 4-bit quantization (MLX handles this natively)
- Sequential model loading with garbage collection
- KV cache management (the hard part)
- PyTorch was too memory-hungry; MLX is 3x more efficient on Apple Silicon

### 4. KV Cache Synchronization is the Hard Problem
The algorithm is conceptually simple. The implementation challenge is keeping three separate KV caches in sync:
- On acceptance: extend all caches with the accepted token
- On rejection: roll back the proposer's cache to before the rejected tokens
- Getting this wrong produces garbage output (we learned this the hard way)

### 5. The Paper's Model Size Ratios Matter
- 0.5B → 7B (draft → target): Too large a gap, almost no tokens accepted
- 0.5B → 3B: Works, but 3B is already fast on Apple Silicon
- 0.5B → 1.5B → 7B: The qualifier bridges the gap (once cache bug is fixed)

## Project Structure

```
momo-akira/
├── pyramid_mlx.py           # MLX-based 3-model PyramidSD (the active development file)
├── main.py                  # FastAPI server entrypoint (PyTorch, needs work)
├── config.yaml              # Model configuration
├── SPEC_V2.md               # Correct algorithm specification
├── PAPER_BRIEF.md            # PyramidSD paper summary
├── src/momo_akira/
│   ├── decoding/
│   │   ├── engine.py         # PyramidSD orchestrator (PyTorch version)
│   │   ├── divergence.py     # Div() on logit distributions
│   │   ├── cache.py          # KV cache manager
│   │   └── adaptive.py       # EMA threshold controller
│   ├── models/
│   │   ├── draft.py          # Draft model loader
│   │   └── verifier.py       # Verifier model loader
│   └── server.py             # FastAPI proxy
├── tests/                    # 31 tests, 81% coverage
└── docs/
    ├── ARCHITECTURE.md
    ├── ALGORITHM.md
    └── CONFIGURATION.md
```

## Quick Start

```bash
# Install
git clone https://github.com/rdreilly58/momo-akira.git
cd momo-akira
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install mlx-lm

# Run 2-model speculative decoding (works now)
python -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft-model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --num-draft-tokens 4 \
  --prompt "What is 2+2?" \
  --max-tokens 50

# Run 3-model PyramidSD (has cache bug — WIP)
python pyramid_mlx.py "What is a hash table?"

# Run tests
pytest tests/
```

## Next Steps

1. **Fix KV cache rollback** — implement proper cache trim on token rejection
2. **Benchmark properly** — compare baseline vs 2-model vs 3-model with correct implementation
3. **Wrap in FastAPI** — expose as OpenAI-compatible endpoint
4. **Consider contributing to mlx-lm** — add `--qualifier-model` flag for native 3-model support

## References

- [PyramidSD Paper (arxiv:2510.12966)](https://arxiv.org/abs/2510.12966) — Byun et al., NeurIPS SPIGM 2025
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's MLX LLM framework
- [Speculative Decoding (Leviathan et al. 2023)](https://arxiv.org/abs/2211.17192) — Original speculative decoding paper

## License

MIT
