# momo-akira

**Token-level 3-model speculative decoding for Apple Silicon, based on the PyramidSD paper (arxiv:2510.12966).**

## What This Is

An implementation of Pyramid Speculative Decoding — a technique that accelerates LLM inference by using three models (draft → qualifier → target) to verify tokens in parallel. The draft model proposes tokens cheaply, the qualifier filters them, and the target does final verification. Because GPU forward passes process multiple tokens nearly as fast as one, verifying K tokens costs roughly the same as generating 1.

This implementation runs entirely on Apple Silicon (M4 Mac mini, 16GB) using MLX with 4-bit quantized models from the Qwen2.5 family.

## Status: Working ✅ (April 3, 2026)

```
Prompt: "What is a hash table?"

Output: "A hash table is a data structure that stores and retrieves
data based on a key-value pair. The key is used to locate the data,
and the value is the data itself..."
```

| Metric | Value |
|--------|-------|
| **Speed** | **20.2 tok/s** (1.09x over 18.6 baseline) |
| **Draft acceptance** | 58% |
| **Qualifier acceptance** | 76% |
| **Peak memory** | 5.44 GB |
| **Models** | 0.5B → 1.5B → 7B (all Qwen2.5, 4-bit MLX) |

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
              │       ONE forward pass via KV cache            │
              │    3. Per-token: |P_D(x) - P_Q(x)| ≤ τ_Q?    │
              │       Accept token or PSDA fallback            │
              │    4. On rejection: trim() caches, resync      │
              │    5. Repeat until l_Q tokens accumulated      │
              │                                                │
              │  STAGE 2: Qualifier buffer → Target            │
              │    1. Target (7B) verifies all l_Q in ONE pass │
              │    2. Compare P_T vs P_Q (NOT P_D — key!)     │
              │    3. Per-token: |P_Q(x) - P_T(x)| ≤ τ_T?    │
              │    4. On rejection: trim() all 3 caches        │
              │    5. Accepted tokens → final output            │
              └────────────────────────────────────────────────┘
```

## Key Insight from the Paper

In Stage 2, the target compares against the **qualifier's** distributions (P_Q), not the draft's (P_D). Since P_Q is closer to P_T than P_D is, more tokens pass verification. This is the whole reason the qualifier model exists — it bridges the distributional gap between the small draft and large target.

## Quick Start

```bash
# Clone and install
git clone https://github.com/rdreilly58/momo-akira.git
cd momo-akira
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install mlx-lm

# Run PyramidSD (downloads models on first run)
python pyramid_mlx.py "Explain how a hash table works."

# Run 2-model speculative decoding (mlx-lm built-in, for comparison)
python -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --draft-model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --num-draft-tokens 4 \
  --prompt "Explain how a hash table works." \
  --max-tokens 150

# Run tests
pytest tests/
```

## How It Works

### Token-Level Speculation (Not Response-Level Routing)

Speculative decoding is NOT "generate a response, score it, try again with a bigger model." It operates at the **token level**:

1. The draft model (0.5B, very fast) generates K candidate tokens sequentially
2. The qualifier model (1.5B) verifies all K tokens in a **single** forward pass by comparing per-token probability distributions
3. Accepted tokens pass to Stage 2; rejected tokens get replaced by the qualifier's prediction (PSDA)
4. The target model (7B) verifies the qualifier-approved buffer in another single forward pass
5. The speedup comes from verifying K tokens in the time it takes to generate 1

### KV Cache Management

The hardest implementation challenge. All three models maintain separate KV caches that must stay synchronized:

- **On acceptance:** caches naturally advance together
- **On rejection:** `trim()` rolls back rejected tokens from all affected caches
- **Cache invariant:** all three caches must be at the same offset at the start of each outer loop iteration
- **Off-by-one fix:** each model's logit at position `i` predicts position `i+1`, so to assess token `i` we use logits from position `i-1` (or the pre-batch logits for `i=0`)

### Softmax Probability Comparison

Raw logit values differ in scale between model sizes (a 0.5B model might assign logit=12.5 to a token while the 7B assigns logit=8.3 for the same prediction). We compare **softmax probabilities** instead, which normalizes to [0,1]:

```
Accept if: |softmax(P_Q)[token] - softmax(P_D)[token]| ≤ τ_Q
```

## Benchmarks

| Mode | Speed | Memory | Notes |
|------|-------|--------|-------|
| 7B baseline (no speculation) | 18.6 tok/s | 4.4 GB | Standard autoregressive |
| 2-model (0.5B→7B, mlx-lm) | 25.0 tok/s | 4.7 GB | Built-in, large distributional gap limits benefit |
| 2-model (0.5B→3B, mlx-lm) | 45.9-67.6 tok/s | 2.1 GB | 3B is fast enough that speculation overhead hurts |
| **3-model PyramidSD** | **20.2 tok/s** | **5.4 GB** | **0.5B→1.5B→7B, correct output** |

The 3-model PyramidSD shows modest speedup (1.09x) over baseline. The paper reports up to 1.91x on GPU — our implementation has significant room for optimization (see Roadmap).

## Improvement Roadmap

Based on comprehensive research survey (40+ papers, April 2026):

### Priority 1: Correctness & Quality
- **Quality validation:** BLEU/exact-match comparison against baseline on CSQA benchmark
- **Lossless rejection sampling:** `accept if random() < min(1, target_prob / draft_prob)` — guarantees output distribution identical to target model (current fuzzy approach trades quality for speed)

### Priority 2: Speed (Biggest Impact)
- **Adaptive draft length (PEARL, ICLR 2025):** Dynamically adjust `l_d` based on draft model entropy — stop drafting early when uncertain, draft longer when confident. Expected +20-40% speed.
- **Token tree speculation (SpecInfer, DySpec, Talon):** Draft a tree of top-k candidates instead of a single chain, verify all branches in one forward pass via tree attention. Expected +50-200% speed.
- **Self-speculative decoding (Apple ReDrafter):** Use the target model's own early layers as the draft, eliminating the separate 0.5B model. 2-2.5x speedup demonstrated on MLX.

### Priority 3: Memory & Cache
- **KV cache quantization (KVSplit, QuantSpec):** 8-bit keys + 4-bit values reduces cache memory by 59% with <1% quality loss. Apple's CommVQ achieves 87.5% reduction.
- **Rotating KV cache:** Cap cache size with `max_kv_size` to bound memory regardless of sequence length. Already supported by mlx-lm.

### Priority 4: Architecture
- **PSDF variant:** Implement the paper's fuzzy variant (both stages use fuzzy acceptance). Higher peak speed (1.91x) but more variance than current PSDA.
- **Benchmark suite:** Systematic sweep of `l_d`, `l_q`, `tau_q`, `tau_t` to find optimal hyperparameters.
- **vllm-mlx integration:** Build on the production-grade MLX inference server for batching, scheduling, and memory management.

### Priority 5: Production
- **FastAPI server:** OpenAI-compatible `/v1/chat/completions` endpoint wrapping the MLX engine
- **Streaming output:** Yield tokens as they're accepted for interactive use
- **Prompt caching:** Reuse cached system prompts across requests

## Project Structure

```
momo-akira/
├── pyramid_mlx.py              # ★ Main implementation (MLX, working)
├── main.py                     # FastAPI server entrypoint (needs MLX wiring)
├── config.yaml                 # Model configuration
├── SPEC_V2.md                  # Algorithm specification
├── KV_CACHE_SPEC.txt           # KV cache implementation spec
├── PAPER_BRIEF.md              # PyramidSD paper summary
├── src/momo_akira/             # PyTorch implementation (reference, OOMs on 16GB)
│   ├── decoding/
│   │   ├── engine.py           # PyramidSD orchestrator
│   │   ├── divergence.py       # Div() on logit distributions
│   │   ├── cache.py            # KV cache manager
│   │   └── adaptive.py         # EMA threshold controller
│   ├── models/
│   │   ├── draft.py            # Draft model loader
│   │   └── verifier.py         # Verifier model loader
│   └── server.py               # FastAPI proxy
├── tests/                      # 31 tests, 81% coverage
│   ├── unit/                   # divergence, cache, config, adaptive, server
│   └── integration/            # full decoding loop
└── docs/
    ├── ARCHITECTURE.md
    ├── ALGORITHM.md
    └── CONFIGURATION.md
```

## Lessons Learned

### 1. Speculative Decoding ≠ Model Routing
Our first attempt (v1) built a response-level cascade router — generate a full response from a cheap model, score it with heuristics, escalate if bad. This is fundamentally NOT speculative decoding. The real algorithm operates at the **token level**: draft proposes K tokens, verifier checks them in parallel via logit distribution comparison. The speedup comes from parallel verification, not model selection.

### 2. API Models Can't Do This
Speculative decoding requires direct access to per-token logit distributions and parallel forward passes. Cloud API models (Claude, GPT) don't expose this. This must run on **local models** with a shared tokenizer.

### 3. Memory Management is Critical
On 16GB Apple Silicon, fitting 3 models requires 4-bit quantization (MLX handles this natively). PyTorch was 3x more memory-hungry than MLX for the same models. Sequential loading with garbage collection is essential.

### 4. KV Cache Synchronization is the Hard Problem
The algorithm is conceptually simple. The implementation challenge is keeping three separate KV caches perfectly synchronized through accept/reject cycles. Getting this wrong produces garbage output that still runs at full speed — making the bug hard to detect. The key insight: track the "off-by-one" carefully — model output at position `i` predicts position `i+1`.

### 5. Softmax Normalization is Essential
Raw logit values differ wildly between model sizes. The 0.5B model might assign logit=12.5 while the 7B assigns logit=8.3 to the same token — same prediction, different scale. Comparing raw logits gives <1% acceptance. Comparing softmax probabilities gives 58%.

### 6. The Paper's Model Size Ratios Matter
The qualifier bridges the distributional gap between draft and target. Without it, 0.5B→7B acceptance is poor (the gap is too large). With the 1.5B qualifier in between, the two-stage verification works because P_Q is close to both P_D and P_T.

## References

- [PyramidSD Paper (arxiv:2510.12966)](https://arxiv.org/abs/2510.12966) — Byun et al., NeurIPS SPIGM 2025
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's MLX LLM framework
- [Speculative Decoding (Leviathan et al. 2023)](https://arxiv.org/abs/2211.17192) — Original speculative decoding paper
- [Fuzzy Speculative Decoding (Holsman et al. 2025)](https://arxiv.org/abs/2502.20704) — Relaxed acceptance criteria
- [PEARL (Liu et al. 2024)](https://arxiv.org/abs/2408.11170) — Adaptive draft length
- [SpecInfer (Miao et al. 2024)](https://arxiv.org/abs/2305.09781) — Token tree speculation
- [Apple ReDrafter (2025)](https://machinelearning.apple.com/research/recurrent-drafter) — Self-speculative on MLX
- [QuantSpec (Apple, 2025)](https://machinelearning.apple.com/research/quantspec) — Quantized KV cache for spec dec
- [Speculative Decoding Papers Collection](https://github.com/hemingkx/SpeculativeDecodingPapers) — Comprehensive paper list

## License

MIT
