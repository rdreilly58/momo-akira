# PyramidSD: 3-Model Speculative Decoding
## arxiv:2510.12966 — Accepted at NeurIPS SPIGM 2025

Authors: Sanghyun Byun, Mohanad Odema, Jung Guack, Baisub Lee, Jacob Song, Woo Seong Chung (LG Electronics USA)

## Abstract

Speculative Decoding (SD) accelerates inference in large language models by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, the throughput gains of SD are fundamentally limited by a trade-off between draft model size and token acceptance: smaller draft models generate tokens more quickly but exhibit greater divergence from the target model, resulting in lower acceptance rates and reduced speedups. We introduce Pyramid Speculative Decoding (PyramidSD), an extension of SD that inserts an intermediate qualifier model between the draft and target to bridge the distributional gap in output predictions, allowing smaller model to be used for drafting. This hierarchical decoding strategy improves alignment across models, enabling higher acceptance rates and allowing the use of significantly smaller draft models without sacrificing overall performance. PyramidSD builds on fuzzy acceptance criteria to support relaxed divergence thresholds at each stage, improving throughput. In experiments, PyramidSD achieves up to 1.91x generation speed over standard SD, reaching 124 tokens per second on a consumer GPU (RTX 4090).

## Core Algorithm

### 3-Model Hierarchy
- Draft model (M_D): smallest, fastest — proposes candidate tokens
- Qualifier model (M_Q): intermediate — bridges distributional gap
- Target model (M_T): largest — final verification

Size ordering: |theta_T| > |theta_Q| > |theta_D|

### Fuzzy Acceptance Criteria

Standard SD requires strict equality matching, which makes the qualifier redundant. PyramidSD uses Fuzzy Speculative Decoding (FSD) with relaxed divergence thresholds:

Two-stage criterion:
  Div(P_M_Q(x_t), P_M_D(x_t)) <= tau_Q  AND  Div(P_M_T(x_t), P_M_Q(x_t)) <= tau_T

Where:
- P_M denotes the logit distribution over the next token x_t
- Div is a divergence measure
- tau_Q, tau_T are divergence thresholds

### Throughput Analysis

Standard SD throughput:
  V_SD = beta_T,D * (l_D + 1) / (l_D/V_D + 1/V_T)

PyramidSD first computes effective speed of draft->qualifier stage:
  V_FSD_Q,D = beta_Q,D * (l_D + 1) / (l_D/V_D + 1/V_Q)

Then overall PyramidSD throughput:
  V_PSD = beta_T,Q * (l_Q + 1) / (l_Q/V_FSD_Q,D + 1/V_T)

Key design levers:
- Speculative lengths (l_D, l_Q)
- Acceptance thresholds (tau_Q, tau_T)

### Two Variants

1. PSDF (Fuzzy): Fuzzy acceptance at both stages. Highest peak performance (1.91x) but higher variance.
2. PSDA (Assisted): Uses assisted decoding at qualifier stage — when draft tokens rejected by M_Q, samples from qualifier's distribution instead. More stable, 1.44x improvement.

### Key Finding: tau_Q <= tau_T works best
The qualifier first filters out clear mismatches while passing plausible candidates forward, allowing the target to verify efficiently.

### Optimal Configuration
- l_D should be approximately half of l_Q (ratio 1:2 to 1:3)
- tau_Q, tau_T grid searched over {0.2, 0.3, 0.4, 0.5}
- l_D in {2, 4, 6}, l_Q in {1, 2, 3, 4, 5, 7, 10, 15, 20, 25}

### Entropy Gradient Across Model Scales
- Small models: high entropy, widespread uncertainty
- Medium models: intermediate, sharper distributions
- Large models: low entropy, highly confident predictions
This natural gradient is what makes the 3-tier cascade effective.

### Assisted Decoding Variant (PSDA)
When draft tokens are rejected by M_Q, instead of standard SD probability rule, PSDA samples directly from the qualifier's distribution. This provides a guaranteed quality floor matching M_Q while still accelerating decoding via M_D.

## Experimental Results (Llama 3.2-1B / 3.2-3B / 3.1-8B on RTX 4090)
- Standard SD: baseline
- FSD: ~1.0-1.2x over SD
- PSDA: up to 1.44x over SD (stable)
- PSDF: up to 1.91x over SD (variable)
- Speed: up to 124 tok/s on RTX 4090

## Key Insights for API-Based Implementation

The paper targets local GPU inference, but the core principles translate to API-based cascades:

1. **Entropy gradient**: Cheaper models are uncertain, expensive models are confident. Use this to route.
2. **Fuzzy thresholds**: Don't require perfect agreement — allow relaxed acceptance within divergence bounds.
3. **Two-stage filtering**: Qualifier catches obvious failures before expensive target is invoked.
4. **tau_Q <= tau_T**: Be stricter at the first gate, more relaxed at the final gate.
5. **PSDA for stability**: When the draft fails, fall back to qualifier's output rather than always escalating to target.
6. **Adaptive tuning**: Optimal thresholds vary by task — suggests runtime adaptation.
