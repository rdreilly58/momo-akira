# Configuration Reference

All configuration lives in `config.yaml`.

## `models`

| Key | Description |
|-----|-------------|
| `family` | Model family name (documentation only — ensures you remember all three must share a tokenizer) |
| `draft.model_id` | HuggingFace model ID for the draft (smallest) model |
| `qualifier.model_id` | HuggingFace model ID for the qualifier (mid) model |
| `target.model_id` | HuggingFace model ID for the target (largest) model |
| `*.device` | `"auto"` (detects MPS/CUDA/CPU), `"cuda"`, `"mps"`, `"cpu"` |
| `*.dtype` | `"float16"`, `"bfloat16"`, `"float32"` |

**All three models must share the same tokenizer vocabulary.**  Use models from
the same family (e.g., all Qwen2.5, all Llama-3.x).

## `decoding`

| Key | Default | Description |
|-----|---------|-------------|
| `variant` | `"psda"` | `"psda"` (Assisted) or `"psdf"` (Fuzzy) |
| `speculative_length_d` | `4` | `l_D`: tokens the draft proposes per inner-loop batch |
| `speculative_length_q` | `10` | `l_Q`: qualifier-verified tokens to accumulate before target check |
| `tau_q` | `0.3` | Draft→qualifier divergence threshold |
| `tau_t` | `0.4` | Qualifier→target divergence threshold |
| `temperature` | `0.7` | Sampling temperature for draft generation |
| `max_new_tokens` | `2048` | Maximum output tokens |

**Tuning tips:**
- `tau_q ≤ tau_t` is recommended (paper Section 4)
- `l_D ≈ l_Q / 2` is optimal in most configurations
- Lower τ = stricter = fewer accepted tokens = more quality, less speed
- Higher τ = permissive = more accepted tokens = more speed, potentially lower quality

## `adaptive`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable runtime threshold adjustment |
| `target_draft_acceptance` | `0.70` | Target fraction of draft tokens accepted by qualifier |
| `target_qualifier_acceptance` | `0.80` | Target fraction of qualifier tokens accepted by target |
| `learning_rate` | `0.05` | EMA alpha for acceptance rate smoothing |
| `warmup_tokens` | `500` | Tokens to generate before adapting thresholds |

## `server`

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `"127.0.0.1"` | Bind address |
| `port` | `7780` | Listen port |

## Recommended Model Families

### Apple Silicon (MPS)
```yaml
models:
  family: "llama3"
  draft:
    model_id: "meta-llama/Llama-3.2-1B-Instruct"
  qualifier:
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
  target:
    model_id: "meta-llama/Llama-3.1-8B-Instruct"
```

### CUDA (default, smaller memory footprint)
```yaml
models:
  family: "qwen2.5"
  draft:
    model_id: "Qwen/Qwen2.5-0.5B-Instruct"
  qualifier:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
  target:
    model_id: "Qwen/Qwen2.5-7B-Instruct"
```
