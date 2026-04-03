# Research Findings: Self-Speculative Decoding Approaches

**Date:** April 3, 2026  
**Project:** momo-akira  
**Context:** We implemented three speculative decoding approaches on Apple Silicon M4 Mac mini (16GB). This document captures what we learned, what works, what doesn't, and why.

---

## The Two Approaches We Conflated (Critical Mistake)

We initially treated "Apple's self-speculative decoding" as a single concept. In reality, there are **two fundamentally different techniques** that share superficial similarities:

### 1. Apple's ReDrafter (Recurrent Drafter)
- **Paper:** arxiv:2403.09919 (March 2024, updated Dec 2024)
- **Authors:** Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang, Yi Wang (Apple)
- **Code:** https://github.com/apple/ml-recurrent-drafter
- **NOT early-exit / layer-skipping**
- Uses a **tiny RNN draft head** attached to the LLM's last hidden state
- The RNN is ~1% of LLM parameters
- Employs beam search + dynamic tree attention for candidate deduplication
- Trained via knowledge distillation from the LLM (1.5h on 8×H100)

### 2. Meta's LayerSkip (Self-Speculative Decoding)
- **Paper:** arxiv:2404.16710 (April 2024)
- **Authors:** Meta AI (Elhoushi et al.)
- **Code:** https://github.com/facebookresearch/LayerSkip
- **IS early-exit / layer-skipping**
- Uses the model's own early layers as draft, remaining layers as verifier
- **Requires special training:** layer dropout + early exit loss
- Without this training recipe, early layers produce garbage distributions
- Meta provides pretrained checkpoints (Llama family only)

---

## What We Built and Tested

### Approach A: 3-Model PyramidSD (pyramid_mlx.py)
- **Draft (0.5B)** → **Qualifier (1.5B)** → **Target (7B)**, all Qwen2.5, 4-bit MLX
- Memory: 5.44 GB peak
- Baseline (7B only): 17.8-22.2 tok/s

| Mode | Speed | vs Baseline | Quality (Match) |
|------|-------|-------------|-----------------|
| Fuzzy (chain) | 18.1 tok/s | 1.01x | 13.6% |
| Lossless (chain) | 12.5 tok/s | 0.70x | 17.3% |
| Fuzzy+Tree+Adaptive | 15.3 tok/s | 0.86x | 6.8% |
| Lossless+Tree+Adaptive | 9.6 tok/s | 0.54x | 20.5% |

**Verdict:** No speedup. The overhead of managing 3 models + KV caches exceeds gains.

### Approach B: Naive Early-Exit Self-Spec (self_spec.py)
- Uses 7B model's first N layers as draft, full 28 layers as verifier
- Memory: 4.29 GB peak (saves 1.15 GB vs 3-model)

| Draft Layers | Accept Rate | Speed | vs Baseline |
|-------------|-------------|-------|-------------|
| 14/28 | 3% | 4.8 tok/s | 0.26x |
| 18/28 | 2% | 4.2 tok/s | 0.23x |
| 22/28 | 20% | 5.4 tok/s | 0.29x |
| 24/28 | 25% | 5.7 tok/s | 0.31x |
| 26/28 | 45% | 6.6 tok/s | 0.36x |
| Baseline | — | 18.4 tok/s | 1.0x |

**Verdict:** Catastrophic failure. Without LayerSkip training, early layers' distributions are wildly different from final layers (2-3% acceptance at 14 layers, 45% even at 26/28 layers). Fuzzy mode produces garbage text. Two sequential forward passes (draft + verify) are always slower than one pass on M4.

---

## Why Speculative Decoding Struggles on Apple Silicon

### The fundamental issue: Memory-bound vs Compute-bound

Speculative decoding exploits **compute parallelism**: verifying K tokens in a single GPU forward pass costs roughly the same wall time as generating 1 token, because GPUs have massive parallel compute that's underutilized during autoregressive generation.

On server GPUs (H100, A100):
- Autoregressive generation is **memory-bandwidth bound** (moving weights to compute units)
- The GPU's compute is only ~5-10% utilized during generation
- Speculative decoding uses this idle compute for verification → free speedup

On Apple Silicon (M4 Mac mini, 16GB):
- Unified memory eliminates the memory-transfer bottleneck somewhat
- The ANE/GPU compute is more balanced with memory bandwidth
- There's **less idle compute to exploit**
- Two forward passes (draft + verify) take roughly 2x the time of one pass
- The overhead of KV cache management (trim, rollback, re-feed) involves many `mx.eval()` synchronization points that break MLX's lazy JIT compilation

### Apple's own MLX findings (from the ReDrafter paper)
- M1 Max: **1.37x** speedup (beam width 1, length 4) — narrow beams only
- M2 Ultra: **1.52x** speedup (beam width 3, length 5) — slightly wider
- H100: **2.8x** speedup (beam width 50+) — massive beams
- Key quote: *"optimal speedup is achieved with narrower beams—1 for the M1 Max and 3 for the M2 Ultra"*
- Key insight: *"the presence of array.item() calls causes MLX to break the program down into small subgraphs"* — this is exactly what our code does extensively

---

## ReDrafter: The Viable Path Forward

### Architecture

```
LLM Forward Pass
    ↓
Last Hidden State (h) ─────────────────────────────────┐
    ↓                                                    │
Token Embedding (e_t) ──→ RNN Cell ──→ MLP + softmax ──→│ Draft Token
    ↑                        │                           │
    └── Previous hidden ─────┘                           │
         state (s_{t-1})                                 │
                                                         │
Repeat T times (beam search with width W) ───────────────┘
    ↓
Dynamic Tree Attention (deduplicate shared prefixes)
    ↓
Pack into "packed beam" (30-60% fewer tokens)
    ↓
LLM verifies ALL candidates in ONE forward pass
    ↓
Accept longest valid prefix
```

### RNN Draft Model Details
- **Input:** LLM's last hidden state `h` + token embedding `e_t`
- **Hidden state:** `g_t = [s_t, h]` where `s_t = f(U·s_{t-1} + W·e_t + b)`
- **One RNN layer** (simple, by design)
- **A few MLP layers** with skip connections
- **Shared softmax** layer (parameters don't grow with prediction length T)
- **Very lightweight:** ~1% of LLM parameters

### Training
- Fix LLM parameters, only train RNN drafter
- **Knowledge distillation**: min KL(p_llm || p_draft)
- NOT ground-truth matching — use LLM's own probability distributions
- Distillation gives ~10% better acceptance rate vs ground-truth training
- Training data: ShareGPT dataset
- Training time: **1.5 hours on 8×H100** (or equivalent)

### Key Innovation: Dynamic Tree Attention
- Beam search produces M candidate sequences with shared prefixes
- Example: 3 beams of length 5 = 15 tokens, but after dedup = 8 tokens
- **30-60% compression** in tokens sent to LLM for verification
- Implemented with standard tensor ops (GPU-friendly, no trie traversal)
- The `dedup_prefix` function is 5 lines of tensor code

### MLX-Specific Performance Notes (from paper)
1. **float16 > bfloat16** (consistently faster)
2. **4-bit quantization >> float16** (significantly faster)
3. **Minimize `.item()` calls** — breaks JIT compilation, forces sync
4. **Lazy evaluation is key** — don't force evaluation between steps
5. **Narrow beams on Apple Silicon:** BW=1-3 is optimal (vs 50+ on H100)

---

## Implementation Plan for ReDrafter on Qwen2.5-7B

### Phase 1: Training the RNN Drafter (~2-4 hours)
1. Need access to H100/A100 or equivalent (Apple paper used 8×H100)
2. Alternative: Our AWS GPU instance (pending quota) or Google Colab
3. Alternative: Train locally on M4 with smaller dataset (slower but possible)
4. Use Qwen2.5-7B as base LLM, train RNN drafter with knowledge distillation
5. Dataset: ShareGPT or similar instruction-following dataset

### Phase 2: MLX Inference Implementation (~1-2 days)
1. Port Apple's MLX implementation to work with Qwen2.5-7B
2. Implement beam search with width 1-3 (Apple Silicon optimal)
3. Implement dynamic tree attention
4. **Critical:** Minimize `.item()` calls, keep computation in lazy graph
5. Benchmark against baseline autoregressive

### Phase 3: Integration (~1 day)
1. FastAPI proxy (like current pyramid_mlx.py)
2. OpenClaw gateway integration
3. Benchmark real-world prompts

### Expected Results
- **Realistic target: 1.3-1.5x speedup** on M4 Mac mini
- **Memory overhead: minimal** (~1% extra for RNN drafter)
- **Quality: lossless** (greedy decoding guarantee)

### Open Questions
1. Can we train the RNN drafter locally on M4? (slower but no GPU rental needed)
2. Will Qwen2.5's architecture (GQA, RoPE) affect drafter compatibility?
3. Should we try Apple's pre-trained Vicuna drafter first as proof of concept?
4. Is the 1.3-1.5x speedup worth the training investment?

---

## References

1. **ReDrafter:** Zhang et al., "Recurrent Drafter for Fast Speculative Decoding in Large Language Models," arXiv:2403.09919, 2024. [Paper: papers/redrafter-arxiv-2403.09919.pdf]
2. **LayerSkip:** Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," ACL 2024, arXiv:2404.16710.
3. **PyramidSD:** Ju et al., "Accelerating LLM Inference via Pyramid Speculative Decoding," arXiv:2510.12966, 2025.
4. **EAGLE:** Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," arXiv:2401.15077, 2024.
5. **Medusa:** Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," arXiv:2401.10774, 2024.
6. **Leviathan et al.:** "Fast Inference from Transformers via Speculative Decoding," ICML 2023.
