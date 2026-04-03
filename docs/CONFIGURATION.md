# Configuration Reference: Momo-Akira

This document is the complete reference for `config.yaml`. Every option is documented with its type, default value, valid range, and effect on system behavior.

Configuration is loaded at startup by `src/momo_akira/config.py`. Environment variables override `config.yaml` values using the mapping described in the [Environment Variable Overrides](#environment-variable-overrides) section.

---

## Complete Annotated config.yaml

```yaml
# =============================================================================
# Momo-Akira — config.yaml
# =============================================================================

# -----------------------------------------------------------------------------
# models
#
# Defines the three model tiers. Each tier specifies the model identifier and
# the provider that hosts it. Draft should be the fastest/cheapest model;
# target should be the most capable.
# -----------------------------------------------------------------------------
models:

  draft:
    # Model identifier as expected by the provider API.
    # Type: string
    # Example values:
    #   Anthropic:  "claude-haiku-3-5"
    #   OpenAI:     "gpt-4o-mini"
    #   OpenRouter: "anthropic/claude-3-haiku"
    model_id: "claude-haiku-3-5"

    # Which provider to use for this tier.
    # Type: string
    # Valid values: "anthropic", "openai", "openrouter"
    provider: "anthropic"

    # Maximum tokens to generate for draft responses.
    # Lower values reduce latency on the fast path.
    # Type: integer, default: 1024
    max_tokens: 1024

    # Temperature for draft generation.
    # Type: float [0.0, 2.0], default: 0.3
    # Note: lower temperature improves confidence score stability
    temperature: 0.3

  qualifier:
    model_id: "claude-sonnet-3-5"
    provider: "anthropic"
    max_tokens: 2048
    temperature: 0.5

  target:
    model_id: "claude-opus-4"
    provider: "anthropic"
    max_tokens: 4096
    temperature: 0.7

# -----------------------------------------------------------------------------
# cascade
#
# Controls the speculative decoding cascade behavior: thresholds, variant
# selection, timeouts, and speculative generation lengths.
# -----------------------------------------------------------------------------
cascade:

  # Variant selects the cascade algorithm.
  # Type: string
  # Valid values:
  #   "psda" — PyramidSD Assisted (default). If draft is rejected, qualifier
  #            generates a fresh response. Quality floor = qualifier.
  #   "psdf" — PyramidSD Fuzzy. Draft may be returned after a fuzzy
  #            acceptance check, potentially bypassing qualifier entirely.
  #            Faster but lower quality floor.
  variant: "psda"

  # Draft acceptance threshold. Draft response is returned directly if
  # confidence C_D >= tau_Q. Lower values accept more drafts (cheaper/faster);
  # higher values reject more drafts (better quality).
  #
  # CONSTRAINT: tau_Q <= tau_T (enforced at startup)
  #
  # Type: float [0.0, 1.0], default: 0.65
  tau_Q: 0.65

  # Qualifier acceptance threshold. Qualifier response is returned if
  # confidence C_Q >= tau_T. Requests that fail this gate proceed to target.
  #
  # Type: float [0.0, 1.0], default: 0.80
  tau_T: 0.80

  # Per-tier timeout in milliseconds. If a tier does not respond within the
  # timeout, the cascade treats it as a failure and escalates to the next tier.
  # Set to 0 to disable timeout for a tier (not recommended for production).
  # Type: integer (milliseconds), default values shown below
  timeouts:
    draft_ms: 5000       # 5 seconds for draft
    qualifier_ms: 15000  # 15 seconds for qualifier
    target_ms: 60000     # 60 seconds for target

  # Maximum number of tokens the draft model is expected to generate speculatively
  # before being evaluated. Relevant if using streaming speculation.
  # Type: integer, default: 128
  speculative_length: 128

  # Number of candidate draft tokens to consider during speculative generation.
  # Higher values allow the cascade to consider more alternatives but increase
  # draft model call cost.
  # Type: integer [1, 8], default: 1
  speculative_candidates: 1

# -----------------------------------------------------------------------------
# confidence
#
# Weights for the four signals composing the composite confidence score C.
# Weights are normalized to sum to 1.0 at runtime, so you can specify them
# as relative proportions (e.g., all equal at 1.0 each) or as exact weights
# that already sum to 1.0.
#
# If a signal is unavailable (e.g., logprobs not supported by the provider),
# its weight is redistributed proportionally to the remaining active signals.
# -----------------------------------------------------------------------------
confidence:

  # Weight for prompt complexity analysis signal.
  # High complexity → lower confidence → more escalation.
  # Type: float >= 0.0, default: 0.30
  w_complexity: 0.30

  # Weight for response coherence scoring signal.
  # Hedging, repetition, and structural issues reduce confidence.
  # Type: float >= 0.0, default: 0.30
  w_coherence: 0.30

  # Weight for logprobs signal (per-token log probabilities).
  # Only used when the provider returns logprobs for the response.
  # OpenAI supports this with logprobs=true; Anthropic does not currently.
  # Set to 0.0 to disable even when available.
  # Type: float >= 0.0, default: 0.40
  w_logprobs: 0.40

  # Weight for self-score signal (model rates its own response).
  # Expensive: requires an extra API call per evaluated response.
  # Set > 0.0 to enable; typically 0.10–0.20 when used.
  # Type: float >= 0.0, default: 0.0 (disabled)
  w_self_score: 0.00

  # Calibration constants for the logprobs signal normalization.
  # These tune the sigmoid mapping from mean log probability to [0, 1].
  # Adjust if your models' typical log probability ranges differ from defaults.
  # Type: float
  logprob_shift: 2.5    # shifts the sigmoid center
  logprob_scale: 0.8    # scales the sigmoid slope

# -----------------------------------------------------------------------------
# adaptive_threshold
#
# When enabled, the adaptive threshold controller observes acceptance rates
# at each tier and adjusts tau_Q and tau_T online to maintain target rates.
#
# The values in cascade.tau_Q and cascade.tau_T serve as initial values when
# adaptive thresholding is enabled.
# -----------------------------------------------------------------------------
adaptive_threshold:

  # Enable or disable adaptive threshold tuning.
  # Type: boolean, default: false
  # Set to true for production deployments with stable query distributions.
  # Leave false for testing or when deterministic behavior is required.
  enabled: false

  # Target fraction of all requests to be answered by the draft tier.
  # The controller raises tau_Q when draft acceptance exceeds this target
  # (to maintain quality) and lowers tau_Q when acceptance falls below it
  # (to improve cost efficiency).
  # Type: float [0.0, 1.0], default: 0.70
  target_draft_acceptance: 0.70

  # Target fraction of qualifier-tier requests (those that reach qualifier)
  # to be answered by the qualifier tier rather than escalated to target.
  # Type: float [0.0, 1.0], default: 0.80
  target_qualifier_acceptance: 0.80

  # Exponential moving average decay factor. Controls how quickly the EMA
  # responds to new observations.
  #   alpha = 0.01 → slow adaptation, stable (good for stable distributions)
  #   alpha = 0.10 → fast adaptation (good for shifting distributions)
  # Type: float (0.0, 1.0), default: 0.05
  ema_alpha: 0.05

  # Step size for threshold adjustments per update. Smaller values result in
  # smoother threshold trajectories; larger values allow faster response to
  # distribution shifts.
  # Type: float, default: 0.005
  adjustment_step: 0.005

  # Hard bounds on tau values enforced by the controller regardless of EMA.
  # tau_Q will never be adjusted below min_tau or above tau_T.
  # tau_T will never be adjusted below tau_Q or above max_tau.
  # Type: float [0.0, 1.0]
  min_tau: 0.30
  max_tau: 0.95

# -----------------------------------------------------------------------------
# server
#
# FastAPI server settings.
# -----------------------------------------------------------------------------
server:

  # Bind address for the proxy server.
  # Type: string, default: "127.0.0.1"
  # Set to "0.0.0.0" to accept external connections.
  host: "127.0.0.1"

  # Port number.
  # Type: integer [1, 65535], default: 7780
  port: 7780

  # Uvicorn log level.
  # Type: string
  # Valid values: "debug", "info", "warning", "error", "critical"
  log_level: "info"

  # Number of Uvicorn worker processes.
  # Type: integer >= 1, default: 1
  # Note: When using adaptive thresholds, set workers: 1 to avoid
  # split-brain threshold state across processes. If you need multiple
  # workers, use an external store for threshold state (not yet implemented).
  workers: 1

  # Whether to include cascade_info metadata in every response.
  # Disable to produce responses indistinguishable from a direct provider call.
  # Type: boolean, default: true
  include_cascade_info: true

  # CORS origins to allow. Empty list disables CORS.
  # Type: list of strings, default: []
  cors_origins: []

# -----------------------------------------------------------------------------
# logging
#
# Structured JSONL request/response logging.
# -----------------------------------------------------------------------------
logging:

  # Directory where log files are written.
  # Files are named openclaw-YYYY-MM-DD.jsonl with daily rotation.
  # Type: string (path), default: "./logs"
  # The directory is created at startup if it does not exist.
  path: "./logs"

  # Log rotation period.
  # Type: string
  # Valid values: "daily", "hourly", "never"
  rotation: "daily"

  # Whether to include full request and response content in log records.
  # Disabling reduces log volume and avoids logging sensitive user content.
  # Type: boolean, default: true
  log_content: true

  # Whether to include full cascade_info in every log record.
  # Type: boolean, default: true
  log_cascade_info: true

  # Maximum size of a single log file before forced rotation (bytes).
  # 0 = no size-based rotation.
  # Type: integer, default: 104857600 (100 MB)
  max_bytes: 104857600

  # Number of rotated log files to retain before deletion.
  # Type: integer, default: 30
  retention_days: 30

# -----------------------------------------------------------------------------
# providers
#
# API key configuration. Keys are always read from environment variables;
# never hardcode API keys in config.yaml.
#
# The values here are the names of the environment variables that hold the
# actual keys. The server reads os.environ[key_env_var] at startup.
# -----------------------------------------------------------------------------
providers:

  anthropic:
    # Environment variable containing the Anthropic API key.
    # Type: string (env var name), default: "ANTHROPIC_API_KEY"
    key_env_var: "ANTHROPIC_API_KEY"

    # Anthropic API base URL. Override for enterprise or proxied endpoints.
    # Type: string (URL), default: "https://api.anthropic.com"
    base_url: "https://api.anthropic.com"

    # Maximum concurrent requests to Anthropic API.
    # Type: integer >= 1, default: 10
    max_concurrency: 10

  openai:
    # Environment variable containing the OpenAI API key.
    # Type: string (env var name), default: "OPENAI_API_KEY"
    key_env_var: "OPENAI_API_KEY"

    # OpenAI API base URL. Override to use Azure OpenAI or a local proxy.
    # Type: string (URL), default: "https://api.openai.com/v1"
    base_url: "https://api.openai.com/v1"

    # Whether to request logprobs from OpenAI completions.
    # Enables the logprobs confidence signal when using OpenAI models.
    # Adds a small amount of response overhead; recommended to leave enabled.
    # Type: boolean, default: true
    request_logprobs: true

    # Number of top token logprobs to request (OpenAI logprobs parameter).
    # Type: integer [1, 20], default: 5
    logprobs_top_k: 5

    max_concurrency: 10

  openrouter:
    # Environment variable containing the OpenRouter API key.
    # Type: string (env var name), default: "OPENROUTER_API_KEY"
    key_env_var: "OPENROUTER_API_KEY"

    # OpenRouter API base URL.
    # Type: string (URL), default: "https://openrouter.ai/api/v1"
    base_url: "https://openrouter.ai/api/v1"

    # HTTP Referer header sent to OpenRouter (used for request attribution).
    # Type: string, default: ""
    http_referer: ""

    max_concurrency: 10
```

---

## Environment Variable Overrides

Any `config.yaml` value can be overridden with an environment variable. The mapping uses double-underscore (`__`) as a nesting separator and the prefix `OPENCLAW_`:

| Environment Variable | Overrides |
|----------------------|-----------|
| `OPENCLAW_SERVER__HOST` | `server.host` |
| `OPENCLAW_SERVER__PORT` | `server.port` |
| `OPENCLAW_CASCADE__TAU_Q` | `cascade.tau_Q` |
| `OPENCLAW_CASCADE__TAU_T` | `cascade.tau_T` |
| `OPENCLAW_CASCADE__VARIANT` | `cascade.variant` |
| `OPENCLAW_MODELS__DRAFT__MODEL_ID` | `models.draft.model_id` |
| `OPENCLAW_MODELS__QUALIFIER__MODEL_ID` | `models.qualifier.model_id` |
| `OPENCLAW_MODELS__TARGET__MODEL_ID` | `models.target.model_id` |
| `OPENCLAW_LOGGING__PATH` | `logging.path` |
| `OPENCLAW_ADAPTIVE_THRESHOLD__ENABLED` | `adaptive_threshold.enabled` |

Provider API keys use the `key_env_var` indirection and are not set via the `OPENCLAW_` prefix. For example, if `providers.anthropic.key_env_var` is `"ANTHROPIC_API_KEY"`, the server reads `os.environ["ANTHROPIC_API_KEY"]` directly.

---

## Configuration Validation

The following constraints are validated at startup. The server will refuse to start if any constraint is violated.

| Constraint | Error |
|------------|-------|
| `cascade.tau_Q <= cascade.tau_T` | `tau_Q must be <= tau_T` |
| All model `provider` values are one of `anthropic`, `openai`, `openrouter` | `Unknown provider` |
| At least one provider API key is set in the environment | `No API keys configured` |
| `cascade.variant` is `psda` or `psdf` | `Unknown variant` |
| `server.workers == 1` when `adaptive_threshold.enabled == true` | Warning (not fatal) |
| All confidence weights are non-negative | `Negative confidence weight` |
| `adaptive_threshold.min_tau < adaptive_threshold.max_tau` | `min_tau must be < max_tau` |

---

## Minimal Production config.yaml

The following is the minimal configuration needed for a production deployment with Anthropic models and default settings:

```yaml
models:
  draft:
    model_id: "claude-haiku-3-5"
    provider: "anthropic"
  qualifier:
    model_id: "claude-sonnet-3-5"
    provider: "anthropic"
  target:
    model_id: "claude-opus-4"
    provider: "anthropic"

cascade:
  variant: "psda"
  tau_Q: 0.65
  tau_T: 0.80

server:
  host: "127.0.0.1"
  port: 7780

providers:
  anthropic:
    key_env_var: "ANTHROPIC_API_KEY"
```

All other values take the defaults documented above.
