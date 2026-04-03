# momo-akira v2 — Correct Specification

## What This Document Is

This is the design specification for a faithful implementation of PyramidSD
(arxiv:2510.12966). It will be used to spawn a Claude Code session that builds
the architecture and implementation from scratch.

The previous v1 implementation was **incorrect** — it built a response-level
cascade router instead of a token-level speculative decoder. This spec corrects
that fundamental misunderstanding.

---

## 1. What Speculative Decoding Actually Is

### The Core Idea

Standard autoregressive decoding generates one token at a time. Each token
requires a full forward pass through the model. For a large model (e.g., 8B
parameters), each forward pass takes ~100ms. Generating 100 tokens = 10 seconds.

Speculative decoding exploits a key GPU property: **a forward pass that
processes 5 tokens costs roughly the same wall-clock time as a forward pass
that processes 1 token.** This is because GPU compute is parallel — the
bottleneck is memory bandwidth, not arithmetic.

The algorithm:
1. A small, fast **draft model** generates K tokens sequentially (K forward
   passes, but each is very fast because the model is small)
2. The large **target model** processes all K tokens in a **single** forward
   pass, producing probability distributions for each position
3. At each position, compare the draft's chosen token against the target's
   distribution. Accept tokens that match; reject starting from the first
   mismatch.
4. On rejection, resample from the target's distribution at that position.

**The speedup comes from verifying K tokens in the time it takes to generate 1.**

### What momo-akira v1 Got Wrong

v1 generated a **complete response** from the draft model, scored it with
surface heuristics, and decided whether to "accept" the whole response or
"escalate" to a bigger model to regenerate from scratch. This is:
- Not token-level (it's response-level)
- Not parallel verification (it's sequential re-generation)
- Not speculative decoding (it's model routing/selection)

---

## 2. What PyramidSD Adds: The 3-Model Hierarchy

Standard speculative decoding uses 2 models (draft + target). PyramidSD adds
a **qualifier** model in between, creating two nested speculation stages.

### The PyramidSD Algorithm (Step by Step)

**Models:**
- M_D (Draft): smallest, fastest (e.g., Llama 3.2-1B, ~500 tok/s)
- M_Q (Qualifier): intermediate (e.g., Llama 3.2-3B, ~200 tok/s)
- M_T (Target): largest, slowest (e.g., Llama 3.1-8B, ~65 tok/s)

All three models MUST share the same tokenizer and vocabulary.

**Parameters:**
- ℓ_D: draft speculative length (how many tokens M_D proposes per batch)
- ℓ_Q: qualifier speculative length (how many verified tokens to accumulate
  before sending to M_T)
- τ_Q: divergence threshold for draft→qualifier acceptance
- τ_T: divergence threshold for qualifier→target acceptance

**The Decoding Loop:**

```
Outer loop (runs until generation is complete):
│
├── STAGE 1: Draft → Qualifier (inner loop)
│   │
│   │  Repeat until ℓ_Q tokens are accepted by M_Q:
│   │   1. M_D generates ℓ_D candidate tokens sequentially
│   │      (ℓ_D fast forward passes)
│   │   2. M_Q processes all ℓ_D tokens in ONE forward pass
│   │      → produces logit distributions P_Q(x_t) for each position
│   │   3. For each token position t = 1..ℓ_D:
│   │      - Compute Div(P_Q(x_t), P_D(x_t))
│   │      - If Div ≤ τ_Q: ACCEPT this token, continue to next
│   │      - If Div > τ_Q: REJECT this token and all subsequent tokens
│   │        → In PSDF: resample from adjusted distribution
│   │        → In PSDA: sample from M_Q's distribution instead
│   │      - Stop checking remaining tokens after first rejection
│   │   4. Accepted tokens are appended to the qualifier's buffer
│   │   5. If fewer than ℓ_Q tokens accumulated, go back to step 1
│   │      with updated context
│   │
│   └── Output: ℓ_Q verified tokens (verified by M_Q)
│
├── STAGE 2: Qualifier → Target
│   │  1. M_T processes all ℓ_Q tokens in ONE forward pass
│   │     → produces logit distributions P_T(x_t) for each position
│   │  2. For each token position t = 1..ℓ_Q:
│   │     - Compare P_T(x_t) against P_Q(x_t) (NOT P_D!)
│   │       This is key: the target compares against the qualifier's
│   │       distribution, which is closer to its own
│   │     - Compute Div(P_T(x_t), P_Q(x_t))
│   │     - If Div ≤ τ_T: ACCEPT
│   │     - If Div > τ_T: REJECT and resample from M_T
│   │     - Stop after first rejection
│   │  3. All accepted tokens are final output
│   │
│   └── Output: accepted tokens added to final sequence
│
└── Continue outer loop with updated context
```

### The Divergence Function: Div()

The paper uses fuzzy speculative decoding (FSD) where:

```
Div(P_A(x_t), P_B(x_t)) = |logit_A(x_t) - logit_B(x_t)|
```

This compares the **logit values** (pre-softmax scores) that each model
assigns to the specific token x_t that was chosen. If the logits are close,
the models "agree" on that token. The threshold τ controls how much
disagreement is tolerated.

**Critical: This requires access to per-token logit distributions, not just
the final text output.**

### Why the Qualifier's Distribution Replaces the Draft's

In Stage 2, the target compares against P_Q (qualifier), NOT P_D (draft).
This is the key insight: P_Q is closer to P_T than P_D is, so the
divergence is smaller, and more tokens pass verification. Without this
substitution, the qualifier adds no value.

### The Two Variants

**PSDF (Fuzzy):** Both stages use fuzzy acceptance with thresholds τ_Q and τ_T.
On rejection, resample from the verifier's adjusted distribution. Achieves up
to 1.91x speedup but has higher variance.

**PSDA (Assisted):** Stage 1 uses assisted decoding — when a draft token is
rejected by M_Q, instead of the standard SD resampling rule, directly sample
the next token from M_Q's distribution. This provides a quality floor at M_Q
level. More stable (1.44x speedup), lower variance.

### Optimal Hyperparameters (from the paper's experiments)

- τ_Q ≤ τ_T (stricter at first gate, more relaxed at second)
- ℓ_D ≈ ℓ_Q / 2 (draft length is roughly half qualifier length)
- Best configs from ablation (PSDF): τ_Q=0.4, τ_T=0.5, ℓ_Q=25, ℓ_D=15
  → 124 tok/s on RTX 4090 (vs 65 tok/s baseline)
- Best configs for PSDA: τ_T=0.5, ℓ_Q=25, ℓ_D=5 → 93.5 tok/s

---

## 3. The Fundamental Challenge: API vs Local Inference

The paper's algorithm operates on **local GPU models** with direct access to:
- Token-level logit distributions (the full vocabulary probability vector)
- Parallel forward passes (process K tokens as cheaply as 1)
- Shared tokenizer across model family
- Sub-millisecond per-token generation from draft models

**API-based LLMs (Claude, GPT, etc.) do NOT provide:**
- Logit distributions (Anthropic exposes nothing; OpenAI exposes top-N logprobs)
- Batch forward passes for verification
- Shared tokenizers across providers
- The latency profile where verification is "free"

### The Implication

A faithful implementation of PyramidSD MUST use **local models** for at least
the draft and qualifier tiers. The target model could potentially be an API
model, but only if we can extract sufficient logprob information for the
divergence comparison.

---

## 4. Implementation Specification for momo-akira v2

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   OpenAI-Compatible API                   │
│              POST /v1/chat/completions                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  PyramidSD Engine                         │
│                                                          │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐          │
│  │  Draft   │───▶│ Qualifier  │───▶│  Target   │          │
│  │ (Local)  │    │  (Local)   │    │(Local/API)│          │
│  │ 0.5-1B   │    │   3B       │    │  7-8B     │          │
│  └─────────┘    └───────────┘    └──────────┘          │
│       │              │               │                   │
│       │    Token-level verification  │                   │
│       │    Logit comparison (Div)    │                   │
│       │    Fuzzy acceptance (τ)      │                   │
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │         Shared Tokenizer                    │         │
│  │    (required: same vocab across all 3)      │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │        KV Cache Manager                     │         │
│  │   (reuse context across verification)       │         │
│  └────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Model Requirements

All three models MUST:
- Share the same tokenizer (e.g., all from the Llama family, or all Qwen)
- Be loadable locally via HuggingFace Transformers or vLLM or MLX
- Expose token-level logit distributions during forward passes

**Recommended baseline model family:**
- Draft: Qwen2.5-0.5B-Instruct (ultra-fast, tiny)
- Qualifier: Qwen2.5-3B-Instruct (good balance)
- Target: Qwen2.5-7B-Instruct (quality ceiling)

Alternative (if targeting Apple Silicon with MLX):
- Draft: Llama-3.2-1B-Instruct
- Qualifier: Llama-3.2-3B-Instruct
- Target: Llama-3.1-8B-Instruct

**Models must be configurable** — swap families by changing config, but they
must always share a tokenizer.

### 4.3 Core Components

**1. Token Generator (Draft)**
- Loads draft model
- Generates ℓ_D tokens autoregressively
- Returns both the token IDs AND the logit distributions at each position
- Must be fast — this runs K sequential forward passes

**2. Token Verifier (Qualifier and Target)**
- Loads verifier model
- Takes a sequence of K candidate tokens
- Runs ONE forward pass over all K tokens simultaneously
- Returns logit distributions at each position
- Computes Div() between its own logits and the proposer's logits
- Returns acceptance mask (which tokens pass, which fail)

**3. Divergence Calculator**
- Implements Div(P_A(x_t), P_B(x_t)) = |logit_A(x_t) - logit_B(x_t)|
- Applies threshold τ to determine accept/reject per token
- Handles the PSDF resampling rule on rejection
- Handles the PSDA assisted fallback on rejection

**4. KV Cache Manager**
- Manages key-value caches for all three models
- On acceptance: extend the cache with accepted tokens
- On rejection: roll back to last accepted position
- Critical for performance — avoids recomputing context

**5. Adaptive Threshold Controller**
- Monitors acceptance rates β_Q and β_T in real time
- Adjusts τ_Q and τ_T to maintain target acceptance rates
- Uses EMA smoothing to avoid oscillation

**6. Orchestrator (PyramidSD Engine)**
- Implements the full two-stage nested loop described in Section 2
- Manages the interplay between draft, qualifier, and target
- Handles end-of-sequence detection
- Collects metrics (tokens/sec, acceptance rates, tier usage)

**7. API Server**
- FastAPI/Flask server exposing /v1/chat/completions
- Accepts OpenAI-format requests
- Converts prompt to tokens using the shared tokenizer
- Runs the PyramidSD engine
- Converts output tokens back to text
- Returns OpenAI-format response with cascade_info metadata

### 4.4 Configuration (config.yaml)

```yaml
models:
  family: "qwen2.5"  # ensures shared tokenizer
  draft:
    model_id: "Qwen/Qwen2.5-0.5B-Instruct"
    device: "auto"           # auto-detect MPS/CUDA/CPU
    dtype: "float16"         # or bfloat16, int8, int4
  qualifier:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    device: "auto"
    dtype: "float16"
  target:
    model_id: "Qwen/Qwen2.5-7B-Instruct"
    device: "auto"
    dtype: "float16"

decoding:
  variant: "psda"            # "psda" or "psdf"
  speculative_length_d: 4    # ℓ_D: tokens drafted per batch
  speculative_length_q: 10   # ℓ_Q: tokens verified before target check
  tau_q: 0.3                 # draft→qualifier divergence threshold
  tau_t: 0.4                 # qualifier→target divergence threshold
  temperature: 0.7
  max_new_tokens: 2048

adaptive:
  enabled: true
  target_draft_acceptance: 0.70
  target_qualifier_acceptance: 0.80
  learning_rate: 0.05
  warmup_tokens: 500

server:
  host: "127.0.0.1"
  port: 7780

logging:
  log_dir: "./logs"
  enabled: true
```

### 4.5 Performance Expectations

On M4 Max (36GB unified memory):
- Draft (0.5B): ~100-200 tok/s generation
- Qualifier (3B): ~30-60 tok/s generation
- Target (7B): ~15-25 tok/s generation
- PyramidSD combined: target 40-60 tok/s (2x over target-only baseline)

Total VRAM: ~6-8 GB for all three models in float16

### 4.6 Testing Requirements

- Unit tests for each component (divergence, verifier, cache, etc.)
- Integration test for the full decoding loop
- Benchmark suite comparing:
  - Target-only baseline (tok/s)
  - Standard 2-model speculative decoding (tok/s)
  - PyramidSD PSDA (tok/s)
  - PyramidSD PSDF (tok/s)
- Quality validation: output from PyramidSD should closely match target-only
  output (measure with BLEU or exact match on deterministic prompts)
- Acceptance rate tracking per stage

### 4.7 Documentation Requirements

- docs/ARCHITECTURE.md: System architecture with diagrams
- docs/DESIGN_DECISIONS.md: Justify choices by referencing the paper
- docs/ALGORITHM.md: Step-by-step algorithm walkthrough
- docs/CONFIGURATION.md: Config reference
- docs/BENCHMARKS.md: Performance results template
- README.md: Quickstart

---

## 5. What Success Looks Like

1. Running `momo-akira serve` starts a local server on port 7780
2. Sending a prompt to /v1/chat/completions generates a response
3. The response was generated using token-level speculative decoding:
   - Draft model proposed tokens
   - Qualifier verified them in parallel
   - Target verified the survivors in parallel
4. Throughput is measurably higher than target-only generation
5. Output quality is equivalent to target-only generation
6. Metrics show per-token acceptance rates for both stages
7. Changing models in config.yaml (within the same family) works without
   code changes

---

## 6. What This Is NOT

- This is NOT a model router that picks which model to use per request
- This is NOT a response-level quality scorer
- This is NOT about API cost optimization (it's about inference speed)
- This is NOT about choosing between Haiku/Sonnet/Opus (those are API models
  without logit access)

This IS a local inference acceleration system that makes a 7B model generate
tokens at 2x speed by using smaller models to speculate and verify.
