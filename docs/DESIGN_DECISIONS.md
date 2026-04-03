# Design Decisions: Momo-Akira

This document captures the rationale behind every major design decision in OpenClaw. Where a decision is grounded in the PyramidSD paper (arxiv:2510.12966, NeurIPS 2025), the relevant section is cited.

---

## 1. Why FastAPI

**Decision:** The proxy is implemented as a FastAPI application rather than Flask, aiohttp, or a raw ASGI app.

**Rationale:**

FastAPI was chosen for three reasons that directly serve this system's requirements:

1. **Native async support.** Speculative decoding cascades involve multiple sequential or conditional LLM API calls. An async-first framework allows the server to handle many in-flight requests concurrently without blocking on I/O, which is critical when a request may touch two or three upstream APIs before returning.

2. **OpenAI compatibility.** The OpenAI Chat Completions API has become the de facto standard for LLM clients. FastAPI's Pydantic-based request/response modeling makes it straightforward to implement exact wire compatibility with the OpenAI schema, including streaming via Server-Sent Events.

3. **Dependency injection.** FastAPI's `Depends()` system provides a clean way to inject the shared `Config` object, the `CascadePipeline`, and the `MetricsCollector` into route handlers without globals, which simplifies testing.

---

## 2. YAML Config as Single Source of Truth

**Decision:** All configuration lives in a single `config.yaml` file, with environment variables allowed to override specific values (primarily secrets and deployment parameters).

**Rationale:**

The cascade has many interacting parameters: two threshold values, six model identifiers, signal weights, provider timeouts, and adaptive threshold hyperparameters. Scattering these across environment variables would make it difficult to reproduce a specific configuration or understand how a deployment is tuned.

`config.yaml` serves as a human-readable specification of the full system state. It can be version-controlled, diffed, and reviewed. Environment variables are reserved for values that must vary between deployments (API keys, host/port) or that are secrets which should not be committed.

The loading order — YAML first, environment variables second — ensures that a production deployment can override any value without modifying the committed config file.

---

## 3. Adapting Div() for API-Based Models

**Decision:** The divergence measure Div(P_i, P_j) from the paper is replaced with a composite confidence score C ∈ [0,1].

**Rationale:**

The PyramidSD paper computes divergence directly between the full token probability distributions of two models. This requires access to per-token logits, which are available when models are running locally but are generally not exposed by commercial API providers.

Specifically, the paper's gating criterion (Section 3.2) is:

```
Accept draft if:  Div(P_Q, P_D) <= tau_Q
Accept qualifier if:  Div(P_T, P_Q) <= tau_T
```

where P_D, P_Q, P_T are the full next-token distributions from draft, qualifier, and target respectively.

Without raw logits, we construct a proxy that captures the same signal — how confident the model is in its response, and how much a larger model would likely disagree — from observable signals:

- **Prompt complexity analysis:** Structural features of the prompt (length, presence of multi-step reasoning markers, code, specialized vocabulary) predict whether simple models will struggle. High-complexity prompts should have lower confidence, pushing toward escalation.

- **Response coherence scoring:** Syntactic and semantic coherence of the draft response. Incoherent or hedging responses signal low model confidence.

- **Logprobs (when available):** OpenAI's API supports `logprobs=true`, which returns per-token log probabilities. When available, these are the strongest direct signal. Mean log probability and entropy over the response tokens approximate the model's distribution certainty.

- **Self-scoring:** The draft model is asked to score its own response on a 0–1 scale with a short follow-up prompt. This is expensive but high-signal and is therefore given a configurable weight that defaults to 0 (disabled) and can be enabled for high-stakes use cases.

The composite score C is a weighted average of these signals, with weights tunable in `config.yaml` under the `confidence` section.

---

## 4. Why Composite Confidence Rather Than a Single Signal

**Decision:** Multiple signals are combined into a weighted composite rather than using a single proxy metric.

**Rationale:**

No single observable signal is a reliable proxy for Div(P_i, P_j) across all query types:

- Logprobs are only available from some providers and some models
- Prompt complexity alone misses cases where a complex-looking prompt has an obvious answer
- Response coherence alone can be gamed by a model that is fluent but factually wrong

The composite approach degrades gracefully: when logprobs are unavailable, their weight can be redistributed to the remaining signals without changing the threshold semantics. The weights are explicitly configurable so operators can tune the scorer for their specific query distribution.

---

## 5. tau_Q <= tau_T Ordering

**Decision:** The configuration system enforces and documents the constraint `tau_Q <= tau_T`.

**Rationale:**

This is a direct finding from the paper (Section 4, Theorem 1 and empirical Section 5.3). The thresholds define a two-level filter: tau_Q is the gate from draft to qualifier, and tau_T is the gate from qualifier to target.

If `tau_Q > tau_T`, a paradox arises: the system would require *higher* confidence to stay at the draft tier than to stay at the qualifier tier. This means queries that are borderline at the draft gate (just below tau_Q) would be escalated to the qualifier, only to immediately pass the qualifier gate (since tau_T < tau_Q). The draft escalation was wasteful.

Setting `tau_Q <= tau_T` creates a monotone filter: each tier requires at least as much confidence as the previous tier to terminate the cascade. The paper shows empirically that this ordering improves both quality and cost efficiency relative to the reversed ordering.

The adaptive threshold controller also respects this constraint: it will never allow tau_Q to rise above tau_T during online tuning.

---

## 6. Why PSDA is the Default Variant

**Decision:** The default cascade variant is PSDA (PyramidSD Assisted) rather than PSDF (PyramidSD Fuzzy).

**Rationale:**

PSDF allows draft responses to be returned after a fuzzy acceptance check at the qualifier stage, without generating a fresh qualifier response. This is faster when the fuzzy check passes, but it means the system can return draft-quality responses on queries that narrowly failed the draft gate.

PSDA maintains a strict quality floor: if the draft is rejected, a fresh qualifier response is always generated. The user is guaranteed a response at least as good as the qualifier model on any query the draft cannot handle confidently.

For a production proxy used in real applications, the stability guarantee of PSDA is more important than the marginal latency savings of PSDF. PSDF is offered as an opt-in variant for operators who have measured their query distribution and determined that the fuzzy acceptance rate is acceptable.

The paper (Section 5.4) also shows that PSDA achieves better quality/cost tradeoffs on most benchmarks, with PSDF only pulling ahead on latency-sensitive workloads where response quality requirements are relaxed.

---

## 7. Adaptive Threshold Controller Design

**Decision:** tau_Q and tau_T are tuned online using an exponential moving average (EMA) of observed tier acceptance rates, targeting a configured acceptance rate per tier.

**Rationale:**

Static thresholds require careful offline tuning and will drift out of calibration as query distributions shift (e.g., a system that starts processing mostly simple queries and later receives more complex ones). An adaptive controller that observes actual acceptance rates and adjusts thresholds to maintain target rates is more robust in production.

The EMA update rule is:

```
acceptance_rate_t = ema_alpha * accepted_this_request + (1 - ema_alpha) * acceptance_rate_{t-1}

if acceptance_rate_t > target_acceptance_rate:
    tau += adjustment_step  # tighten: require more confidence to accept
else:
    tau -= adjustment_step  # relax: accept more easily
```

EMA was chosen over a windowed average because it requires O(1) memory, has a single interpretable hyperparameter (alpha, the decay rate), and gives more weight to recent observations, which is desirable when query distributions are non-stationary.

The controller is optional (disabled by default for deterministic behavior) and can be enabled in `config.yaml` under `adaptive_threshold.enabled`. When disabled, static tau_Q and tau_T from the config file are used.

---

## 8. Provider Abstraction

**Decision:** LLM calls are made through an abstract `LLMProvider` interface with separate implementations for Anthropic, OpenAI, and OpenRouter.

**Rationale:**

The three default model tiers (Haiku, Sonnet, Opus) are Anthropic models, but operators may want to:
- Mix providers (e.g., a cheap OpenAI model as draft, a powerful Anthropic model as target)
- Use OpenRouter to access models not available directly
- Swap in a self-hosted model at any tier

The provider abstraction allows any combination. The interface is minimal — `complete()` and `stream()` — so adding a new provider requires only implementing two methods and a logprobs extraction hook.

OpenRouter is included as a first-class provider because it exposes a broad model catalog under a single OpenAI-compatible endpoint, making it the easiest path to testing the cascade with different model combinations.

---

## 9. JSONL Logging Format

**Decision:** Request/response logs are written in newline-delimited JSON (JSONL) with daily rotation.

**Rationale:**

JSONL is the standard format for structured log data that will be consumed by log aggregation tools (Splunk, Datadog, Elastic, etc.). Each line is a complete, self-contained JSON object, making it trivial to stream-process logs with `jq`, `grep`, or any log pipeline.

Daily rotation balances file size against the overhead of rotation. A high-throughput deployment producing many requests per second will generate log files that are large but manageable over a day; rolling over at midnight ensures that log files align with natural analysis windows (daily cost, daily acceptance rates).

The log schema includes all fields needed for post-hoc analysis without re-querying the original request:
- Full request and response content (for quality auditing)
- `cascade_info` (tier, confidence, latency per tier) for performance analysis
- `cost_usd` for cost attribution
- `request_id` for correlation with application-level logs

Log files are not encrypted by default. Operators handling sensitive data should configure log path to a volume with appropriate access controls, or disable content logging and retain only metadata.
