# Cascade Algorithm: Momo-Akira

This document provides a complete technical description of the speculative decoding cascade, including pseudocode for both variants, the confidence scoring algorithm, divergence measure adaptations, the threshold adaptation algorithm, and edge case handling.

---

## Background

PyramidSD (arxiv:2510.12966) frames speculative decoding as a hierarchical acceptance test over a pyramid of models ordered by quality: Draft < Qualifier < Target. The cascade short-circuits as soon as a tier's response passes the divergence gate, avoiding unnecessary calls to larger, more expensive models.

The central observation from the paper is that token-level divergence between a smaller and larger model is low on easy queries (the models agree), and high on hard queries (the models disagree). By measuring this divergence, the cascade can route easy queries to cheap models and hard queries to accurate ones.

In an API setting, token distributions are not directly accessible. We construct a confidence score C as a proxy for low divergence: high C means the smaller model is likely producing a response similar to what the larger model would produce.

---

## PSDA: PyramidSD Assisted

PSDA guarantees a quality floor equal to the qualifier model. If the draft is rejected, the qualifier generates a completely fresh response. This is the more conservative and stable variant.

### Pseudocode

```
function cascade_psda(request, tau_Q, tau_T):

    # --- TIER 1: DRAFT ---
    t0 = now()
    draft_response = await draft_provider.complete(request)
    draft_latency = now() - t0

    C_D = confidence_score(request, draft_response)

    if C_D >= tau_Q:
        return CascadeResult(
            response   = draft_response,
            tier       = "draft",
            confidence = C_D,
            cost       = draft_provider.cost(request, draft_response),
        )

    # --- TIER 2: QUALIFIER ---
    t1 = now()
    qualifier_response = await qualifier_provider.complete(request)  # fresh call
    qualifier_latency = now() - t1

    C_Q = confidence_score(request, qualifier_response)

    if C_Q >= tau_T:
        return CascadeResult(
            response   = qualifier_response,
            tier       = "qualifier",
            confidence = C_Q,
            cost       = draft_cost + qualifier_provider.cost(request, qualifier_response),
        )

    # --- TIER 3: TARGET ---
    t2 = now()
    target_response = await target_provider.complete(request)
    target_latency = now() - t2

    return CascadeResult(
        response   = target_response,
        tier       = "target",
        confidence = 1.0,  # unconditional
        cost       = draft_cost + qualifier_cost + target_provider.cost(request, target_response),
    )
```

### Properties

- Quality floor = qualifier model on any query that fails the draft gate
- Tier 2 always produces a new response; draft response is discarded on rejection
- Three possible outcomes: draft, qualifier, target
- Tail latency is bounded by `draft_latency + qualifier_latency + target_latency`

---

## PSDF: PyramidSD Fuzzy

PSDF introduces a fuzzy acceptance check at the qualifier stage. If the draft response narrowly failed the draft gate but scores above tau_T on a secondary evaluation, it is returned without generating a qualifier response. This can reduce cost and latency when the draft response is "good enough" even if not confidently so.

### Pseudocode

```
function cascade_psdf(request, tau_Q, tau_T):

    # --- TIER 1: DRAFT ---
    t0 = now()
    draft_response = await draft_provider.complete(request)
    draft_latency = now() - t0

    C_D = confidence_score(request, draft_response)

    if C_D >= tau_Q:
        return CascadeResult(
            response   = draft_response,
            tier       = "draft",
            confidence = C_D,
            cost       = draft_provider.cost(request, draft_response),
        )

    # --- FUZZY GATE: Re-evaluate draft against tau_T ---
    # The draft failed the draft gate (C_D < tau_Q) but may still be
    # acceptable by the softer qualifier standard (tau_T).
    C_fuzzy = fuzzy_confidence_score(request, draft_response)

    if C_fuzzy >= tau_T:
        return CascadeResult(
            response    = draft_response,
            tier        = "draft_fuzzy",
            confidence  = C_fuzzy,
            cost        = draft_provider.cost(request, draft_response),
        )

    # --- TIER 2: QUALIFIER ---
    t1 = now()
    qualifier_response = await qualifier_provider.complete(request)
    qualifier_latency = now() - t1

    C_Q = confidence_score(request, qualifier_response)

    if C_Q >= tau_T:
        return CascadeResult(
            response   = qualifier_response,
            tier       = "qualifier",
            confidence = C_Q,
            cost       = draft_cost + qualifier_provider.cost(request, qualifier_response),
        )

    # --- TIER 3: TARGET ---
    t2 = now()
    target_response = await target_provider.complete(request)

    return CascadeResult(
        response   = target_response,
        tier       = "target",
        confidence = 1.0,
        cost       = draft_cost + qualifier_cost + target_provider.cost(request, target_response),
    )
```

### Properties

- Quality floor is lower than PSDA: draft responses can be returned from the fuzzy gate
- Four possible outcomes: draft, draft_fuzzy, qualifier, target
- Fuzzy gate adds a scoring step but avoids a full qualifier API call when it passes
- Best suited for latency-sensitive workloads with relaxed quality requirements

---

## Confidence Scoring Algorithm

The confidence scorer computes a composite score C ∈ [0,1] for a (request, response) pair. This is the API-level proxy for the paper's divergence measure Div(P_larger, P_smaller).

### Signals

#### 1. Prompt Complexity (w_complexity)

Analyzes the prompt to estimate how likely a small model is to produce a low-quality response.

```
function prompt_complexity(request) -> float [0, 1]:
    score = 0.0
    prompt_text = concat(request.messages)

    # Length penalty: longer prompts are harder
    score += min(len(prompt_text) / MAX_PROMPT_CHARS, 1.0) * 0.2

    # Multi-step reasoning markers
    if contains_reasoning_markers(prompt_text):  # "step by step", "first ... then"
        score += 0.2

    # Code presence
    if contains_code_blocks(prompt_text):
        score += 0.2

    # Specialized vocabulary (domain terms, proper nouns density)
    score += specialized_vocab_density(prompt_text) * 0.2

    # Instruction ambiguity (conflicting or underspecified constraints)
    score += ambiguity_score(prompt_text) * 0.2

    # complexity is high → confidence should be LOW
    return 1.0 - score
```

#### 2. Response Coherence (w_coherence)

Measures the structural and semantic quality of the response text.

```
function response_coherence(response) -> float [0, 1]:
    text = response.content

    # Hedging penalty: "I'm not sure", "I think", "it might be"
    hedge_ratio = count_hedges(text) / max(word_count(text), 1)
    hedge_penalty = min(hedge_ratio * 10, 0.4)

    # Repetition penalty: high n-gram repetition signals confused output
    rep_penalty = repetition_score(text) * 0.3

    # Sentence structure: well-formed sentences vs. fragments
    structure_score = sentence_quality_score(text) * 0.3

    return max(0.0, 1.0 - hedge_penalty - rep_penalty + structure_score)
```

#### 3. Logprobs Signal (w_logprobs)

When the provider returns per-token log probabilities, they are the most direct proxy for model confidence.

```
function logprobs_confidence(response) -> float [0, 1]:
    if response.logprobs is None:
        return None  # signal unavailable; weight is redistributed

    token_logprobs = response.logprobs  # list of log(p) per token

    # Mean log probability (higher is more confident)
    mean_lp = mean(token_logprobs)

    # Entropy estimate from top-k logprobs (lower entropy = more confident)
    entropy = estimate_entropy(response.logprobs_top_k)

    # Normalize mean_lp to [0, 1] using empirical calibration constants
    # A mean_lp near 0.0 means near-certain; near -5.0 means very uncertain
    normalized_lp = sigmoid((mean_lp + LOGPROB_SHIFT) * LOGPROB_SCALE)

    # Entropy term: normalize to [0, 1] and invert
    normalized_entropy = 1.0 - min(entropy / MAX_ENTROPY, 1.0)

    return 0.6 * normalized_lp + 0.4 * normalized_entropy
```

#### 4. Self-Score (w_self_score)

The draft model is asked to rate its own response. This is the most expensive signal and is disabled by default.

```
function self_score(request, response, provider) -> float [0, 1]:
    score_prompt = format_self_score_prompt(request, response)
    score_response = await provider.complete(score_prompt, max_tokens=5)
    return parse_numeric_score(score_response)  # expects "0.85" etc.
```

### Composite Score

```
function confidence_score(request, response) -> float [0, 1]:
    signals = {}
    weights = {}

    signals["complexity"] = prompt_complexity(request)
    weights["complexity"] = config.confidence.w_complexity

    signals["coherence"] = response_coherence(response)
    weights["coherence"] = config.confidence.w_coherence

    lp = logprobs_confidence(response)
    if lp is not None:
        signals["logprobs"] = lp
        weights["logprobs"] = config.confidence.w_logprobs
    else:
        # Redistribute logprobs weight to other signals
        redistribute_weight("logprobs", weights)

    if config.confidence.w_self_score > 0:
        signals["self_score"] = self_score(request, response, provider)
        weights["self_score"] = config.confidence.w_self_score

    total_weight = sum(weights.values())
    C = sum(signals[k] * weights[k] for k in signals) / total_weight
    return C
```

Default weights (configurable):

| Signal | Default Weight | Notes |
|--------|---------------|-------|
| prompt_complexity | 0.30 | Always available |
| response_coherence | 0.30 | Always available |
| logprobs | 0.40 | Available from OpenAI; redistributed if absent |
| self_score | 0.00 | Disabled by default; expensive |

---

## Divergence Measure Adaptations

The paper uses two divergence measures: total variation (TV) distance and KL divergence. Without token distributions, we adapt these for response-level comparison.

### Total Variation Distance Proxy

At the token level, TV distance between distributions P and Q is:

```
TV(P, Q) = 0.5 * sum_i |P(x_i) - Q(x_i)|
```

At the response level, we approximate TV distance between two responses using normalized edit distance:

```
function tv_distance_proxy(response_a, response_b) -> float [0, 1]:
    # Normalize Levenshtein distance by max length
    edit_d = levenshtein(response_a.content, response_b.content)
    max_len = max(len(response_a.content), len(response_b.content), 1)
    return edit_d / max_len
```

This is used when two responses are available (e.g., when comparing draft and qualifier outputs in analysis or in PSDF fuzzy scoring).

### KL Divergence Proxy

KL divergence at the token level is:

```
KL(P || Q) = sum_i P(x_i) * log(P(x_i) / Q(x_i))
```

Without token distributions, we approximate KL divergence using the difference in self-reported log probabilities when available, or vocabulary overlap between responses:

```
function kl_divergence_proxy(response_a, response_b) -> float [0, inf):
    if both responses have logprobs:
        # Use mean logprob difference as asymmetric divergence proxy
        return abs(mean(response_a.logprobs) - mean(response_b.logprobs))
    else:
        # Vocabulary overlap (Jaccard distance on n-gram sets)
        ngrams_a = extract_ngrams(response_a.content, n=3)
        ngrams_b = extract_ngrams(response_b.content, n=3)
        jaccard = len(ngrams_a & ngrams_b) / max(len(ngrams_a | ngrams_b), 1)
        return 1.0 - jaccard  # higher = more divergent
```

---

## Adaptive Threshold Controller

The `AdaptiveThresholdController` maintains online estimates of acceptance rates at the draft and qualifier tiers and adjusts tau_Q and tau_T to hit target acceptance rates.

### Algorithm

```
state:
    tau_Q          = config.cascade.tau_Q         # initial value
    tau_T          = config.cascade.tau_T         # initial value
    ema_draft      = target_draft_acceptance      # EMA of draft acceptance
    ema_qualifier  = target_qualifier_acceptance  # EMA of qualifier acceptance
    alpha          = config.adaptive_threshold.ema_alpha   # e.g., 0.05
    step           = config.adaptive_threshold.adjustment_step  # e.g., 0.01

function update(tier_reached):
    # Update draft acceptance EMA
    draft_accepted = 1 if tier_reached == "draft" else 0
    ema_draft = alpha * draft_accepted + (1 - alpha) * ema_draft

    # Adjust tau_Q based on draft acceptance rate
    if ema_draft > target_draft_acceptance:
        tau_Q = min(tau_Q + step, tau_T)  # tighten: harder to accept at draft
    elif ema_draft < target_draft_acceptance:
        tau_Q = max(tau_Q - step, MIN_TAU)  # relax

    # Update qualifier acceptance EMA (only for requests that reached qualifier)
    if tier_reached in ("qualifier", "target"):
        qualifier_accepted = 1 if tier_reached == "qualifier" else 0
        ema_qualifier = alpha * qualifier_accepted + (1 - alpha) * ema_qualifier

        # Adjust tau_T based on qualifier acceptance rate
        if ema_qualifier > target_qualifier_acceptance:
            tau_T = min(tau_T + step, MAX_TAU)
        elif ema_qualifier < target_qualifier_acceptance:
            tau_T = max(tau_T - step, tau_Q)  # never allow tau_T < tau_Q

function get_thresholds() -> (tau_Q, tau_T):
    return (tau_Q, tau_T)
```

### Invariants

The controller always maintains:
- `MIN_TAU <= tau_Q <= tau_T <= MAX_TAU`
- `tau_Q` adjustments are clamped to `tau_T` (upper bound)
- `tau_T` adjustments are clamped to `tau_Q` (lower bound)

### Typical Target Acceptance Rates

These are starting points; the right values depend on your query distribution and cost/quality requirements:

| Tier | Target Acceptance Rate | Meaning |
|------|----------------------|---------|
| draft | 0.70 | 70% of all requests answered by draft |
| qualifier | 0.80 | 80% of requests that reach qualifier answered there |
| target | 1.00 | Remaining requests go to target unconditionally |

---

## Edge Cases and Graceful Degradation

### Tier Timeouts

Each tier has a configurable timeout (`cascade.timeouts.draft_ms`, `cascade.timeouts.qualifier_ms`, `cascade.timeouts.target_ms`). If a tier times out:

```
if draft times out:
    log warning, set C_D = 0.0, proceed to qualifier
    (treat timeout as rejection, do not return partial response)

if qualifier times out:
    log warning, set C_Q = 0.0, proceed to target

if target times out:
    log error
    if last_successful_response is not None:
        return last_successful_response with tier="degraded"
    else:
        return 503 Service Unavailable
```

### API Failures

Provider API errors (rate limits, 5xx, authentication failure) are caught and handled per tier:

```
if draft API fails:
    log error with provider and status code
    skip draft, proceed directly to qualifier
    (do not count as draft rejection for threshold adaptation)

if qualifier API fails:
    log error
    skip qualifier, proceed directly to target

if target API fails:
    if qualifier_response is not None:
        return qualifier_response with tier="degraded"
    elif draft_response is not None:
        return draft_response with tier="degraded"
    else:
        return 503
```

The `cascade_info.tier` field in the response will be `"degraded"` whenever a fallback path was used, allowing downstream systems to detect degraded operation.

### Confidence Scorer Failures

If the confidence scorer itself raises an exception (e.g., a malformed response that breaks the coherence scorer), the scorer falls back to a pessimistic default:

```
try:
    C = confidence_score(request, response)
except Exception as e:
    log warning(f"Confidence scorer failed: {e}, defaulting to C=0.0")
    C = 0.0  # force escalation to next tier
```

This ensures scorer bugs result in higher-quality (more expensive) responses rather than confidently returning bad draft responses.

### Empty or Malformed Responses

If a model returns an empty response or a response that cannot be parsed:

```
if response.content is None or len(response.content.strip()) == 0:
    C = 0.0  # treat as minimum confidence, force escalation
```

---

## Performance Analysis: When Each Tier Gets Invoked

The fraction of requests reaching each tier depends on tau_Q, tau_T, and the query distribution. The following analysis assumes a typical general-purpose assistant workload.

### Expected Tier Distribution

```
tau_Q = 0.65, tau_T = 0.80 (example values)

Tier         Fraction    Cost multiplier    Quality
─────────────────────────────────────────────────────
draft        ~60-75%     1x                 draft
qualifier    ~15-25%     ~3x                qualifier
target       ~5-15%      ~10x               target

Weighted average cost ≈ 1.0 * 0.70 + 3.0 * 0.20 + 10.0 * 0.10
                      ≈ 0.70 + 0.60 + 1.00
                      ≈ 2.30x draft cost
                      ≈ 0.23x target cost (4.3x savings vs. always using target)
```

### Query Characteristics That Favor Each Tier

**Draft tier is sufficient when:**
- Short, factual questions with unambiguous answers
- Simple summarization of short text
- Format conversion tasks (JSON, markdown, etc.)
- Queries where the draft model produces high logprob, low-entropy responses

**Qualifier tier is invoked when:**
- Multi-step reasoning is required but not highly specialized
- Moderate-length document analysis
- Code generation for common patterns
- Draft response contains hedging language or structural issues

**Target tier is invoked when:**
- Deep domain expertise required (medical, legal, advanced technical)
- Long-form content requiring sustained coherence
- Complex multi-constraint optimization
- High-stakes decisions where quality requirements override cost
- All queries if both draft and qualifier fail their gates
