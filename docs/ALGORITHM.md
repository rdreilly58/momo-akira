# PyramidSD Algorithm Walkthrough

This document describes the token-level decoding algorithm implemented in
`src/momo_akira/decoding/engine.py`, following arxiv:2510.12966.

## Models

| Tier      | Role       | Example             | Speed   |
|-----------|------------|---------------------|---------|
| M_D       | Draft      | Qwen2.5-0.5B        | ~150 tok/s |
| M_Q       | Qualifier  | Qwen2.5-3B          | ~45 tok/s  |
| M_T       | Target     | Qwen2.5-7B          | ~20 tok/s  |

All three models **must share the same tokenizer** (same vocabulary).

## The Divergence Function

```
Div(P_A(x_t), P_B(x_t)) = |logit_A(x_t) - logit_B(x_t)|
```

Compares the raw (pre-softmax) logit that each model assigns to the specific
token `x_t` that was chosen.  If `Div ≤ τ`, the token is accepted.

## The Full Loop

```
Outer loop (until max_new_tokens or EOS):
│
├── STAGE 1: Draft → Qualifier  [inner loop]
│   │
│   │  Repeat until l_Q tokens are accepted by M_Q:
│   │   1. M_D generates l_D tokens sequentially
│   │      (l_D fast forward passes through the tiny draft model)
│   │   2. M_Q processes all l_D tokens in ONE forward pass
│   │      → logit distributions P_Q(x_t) for each position t
│   │   3. For t = 1..l_D:
│   │      - Compute Div(P_Q(x_t), P_D(x_t))
│   │      - If Div ≤ τ_Q: ACCEPT, continue to t+1
│   │      - If Div > τ_Q: REJECT, compute replacement from M_Q, stop
│   │   4. Accepted tokens go into the qualifier buffer
│   │   5. If buffer < l_Q tokens, run inner loop again
│   │
│   └── Output: l_Q tokens verified by M_Q
│
├── STAGE 2: Qualifier buffer → Target
│   │  1. M_T processes all l_Q buffer tokens in ONE forward pass
│   │     → logit distributions P_T(x_t) for each position t
│   │  2. For t = 1..l_Q:
│   │     - Compute Div(P_T(x_t), P_Q(x_t))   ← compares against QUALIFIER, not draft
│   │     - If Div ≤ τ_T: ACCEPT
│   │     - If Div > τ_T: REJECT, resample from M_T, stop
│   │  3. Accepted tokens are final output
│   │
│   └── Output: accepted final tokens
│
└── Append to output sequence, update KV caches, continue
```

## Why Stage 2 Compares Against Qualifier

In Stage 2, the target model compares its distribution against P_Q
(qualifier), not P_D (draft).  P_Q is closer to P_T than P_D is.
This means the divergence is smaller, more tokens pass verification,
and the qualifier tier adds real value.  If Stage 2 compared against P_D,
the qualifier would provide no benefit over 2-model standard speculative decoding.

## Acceptance Variants

### PSDA (Pyramid Speculative Decoding — Assisted)

On rejection at position t:
- Take the next token directly from M_Q's distribution (greedy argmax)
- Quality floor guaranteed at M_Q level
- More stable, lower variance in speedup (~1.44x)

### PSDF (Pyramid Speculative Decoding — Fuzzy)

On rejection at position t:
- Resample from adjusted distribution: `max(0, P_Q - P_D)` normalized
- Theoretically correct per speculative decoding theory
- Higher peak speedup (~1.91x) but more variable

## Hyperparameter Guidance

From the paper's ablation:
- `τ_Q ≤ τ_T` (be stricter at the first gate)
- `l_D ≈ l_Q / 2` (draft length is half the qualifier target length)
- Best PSDF config: `τ_Q=0.4, τ_T=0.5, l_Q=25, l_D=15` → 124 tok/s (RTX 4090)
- Best PSDA config: `τ_T=0.5, l_Q=25, l_D=5` → 93.5 tok/s

Our defaults (smaller models, Apple Silicon target):
```yaml
speculative_length_d: 4   # l_D
speculative_length_q: 10  # l_Q
tau_q: 0.3
tau_t: 0.4
```
