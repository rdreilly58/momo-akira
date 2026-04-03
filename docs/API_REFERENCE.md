# API Reference: OpenClaw Speculative Decoder

The OpenClaw proxy exposes an OpenAI-compatible HTTP API on `localhost:7780` by default. Any client that works with the OpenAI SDK or the Chat Completions REST API will work with OpenClaw without modification.

OpenClaw extends the standard OpenAI response format with a `cascade_info` field that provides observability into which tier handled the request and at what cost.

---

## Base URL

```
http://localhost:7780
```

Override host and port in `config.yaml` under `server.host` and `server.port`.

---

## Authentication

The proxy does not require authentication from the client. It authenticates to upstream providers using the API keys configured via environment variables. If you need to add client authentication, place a reverse proxy (nginx, Caddy) in front of OpenClaw.

---

## Endpoints

### POST /v1/chat/completions

Generate a chat completion using the speculative decoding cascade. This endpoint is a superset of the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).

#### Request

**Headers**

| Header | Value | Required |
|--------|-------|----------|
| `Content-Type` | `application/json` | Yes |

**Body**

```json
{
  "model": "string",
  "messages": [
    {
      "role": "system" | "user" | "assistant",
      "content": "string"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "stream": false,
  "top_p": 1.0,
  "n": 1,
  "stop": null,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "user": "string"
}
```

**Field Descriptions**

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `model` | string | Yes | — | Model name is accepted for compatibility but the cascade selects the actual model. Pass any string (e.g., `"openclaw"`, `"auto"`, or your preferred target model name). |
| `messages` | array | Yes | — | List of message objects with `role` and `content`. Supports system, user, and assistant roles. |
| `max_tokens` | integer | No | per-tier config | Maximum tokens to generate. Passed to the winning tier's model. |
| `temperature` | float | No | per-tier config | Sampling temperature [0.0, 2.0]. Overrides the per-tier default from config. |
| `stream` | boolean | No | `false` | If `true`, returns a stream of Server-Sent Events (SSE) in OpenAI streaming format. `cascade_info` is sent as the final SSE chunk. |
| `top_p` | float | No | `1.0` | Nucleus sampling parameter. Passed through to the winning tier's model. |
| `n` | integer | No | `1` | Number of completions. Only `n=1` is supported; values > 1 are rejected with 400. |
| `stop` | string or array | No | `null` | Stop sequences. Passed through to the provider. |
| `presence_penalty` | float | No | `0.0` | Presence penalty [-2.0, 2.0]. Passed through where supported. |
| `frequency_penalty` | float | No | `0.0` | Frequency penalty [-2.0, 2.0]. Passed through where supported. |
| `user` | string | No | `null` | End-user identifier. Logged but not forwarded to providers. |

#### Response (non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "claude-haiku-3-5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 9,
    "total_tokens": 37
  },
  "cascade_info": {
    "tier": "draft",
    "confidence": 0.91,
    "variant": "psda",
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "draft_model": "claude-haiku-3-5",
    "qualifier_model": "claude-sonnet-3-5",
    "target_model": "claude-opus-4",
    "draft_ms": 342,
    "qualifier_ms": null,
    "target_ms": null,
    "total_ms": 342,
    "cost_usd": 0.000021,
    "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
  }
}
```

**`cascade_info` Fields**

| Field | Type | Description |
|-------|------|-------------|
| `tier` | string | Which tier produced the final response: `"draft"`, `"draft_fuzzy"` (PSDF only), `"qualifier"`, `"target"`, or `"degraded"` (fallback path after failure). |
| `confidence` | float | Confidence score of the winning tier's response [0.0, 1.0]. |
| `variant` | string | Cascade variant used: `"psda"` or `"psdf"`. |
| `tau_Q` | float | Draft acceptance threshold at time of request. |
| `tau_T` | float | Qualifier acceptance threshold at time of request. |
| `draft_model` | string | Model ID of the draft tier. |
| `qualifier_model` | string | Model ID of the qualifier tier. |
| `target_model` | string | Model ID of the target tier. |
| `draft_ms` | integer or null | Latency of the draft tier call in milliseconds. `null` if not called. |
| `qualifier_ms` | integer or null | Latency of the qualifier tier call. `null` if not called. |
| `target_ms` | integer or null | Latency of the target tier call. `null` if not called. |
| `total_ms` | integer | Total end-to-end latency including all tier calls and scoring. |
| `cost_usd` | float | Estimated cost of this request in USD based on token usage and provider pricing. |
| `request_id` | string | UUID for this request, included in access logs for correlation. |

`cascade_info` is omitted when `server.include_cascade_info: false` is set in config.

#### Response (streaming)

When `"stream": true`, the response is a Server-Sent Event stream. Each chunk follows the standard OpenAI streaming format. The final chunk before `[DONE]` includes `cascade_info`:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"claude-haiku-3-5","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"claude-haiku-3-5","choices":[{"index":0,"delta":{"content":"The capital"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"claude-haiku-3-5","choices":[{"index":0,"delta":{"content":" of France is Paris."},"finish_reason":"stop"}],"cascade_info":{"tier":"draft","confidence":0.91,"draft_ms":342,"total_ms":342,"cost_usd":0.000021,"request_id":"f47ac10b-58cc-4372-a567-0e02b2c3d479"}}

data: [DONE]
```

Note: When streaming is enabled, tier selection still happens before streaming begins. The cascade determines which tier to use, then streams the response from that tier. This means streaming latency includes any tier-switching overhead before the first token appears.

#### Error Responses

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `invalid_request_error` | Missing required fields, invalid parameter values, or `n > 1`. |
| 503 | `service_unavailable` | All tiers failed (timeouts or API errors). |
| 504 | `gateway_timeout` | Target tier timed out and no fallback response is available. |

```json
{
  "error": {
    "message": "All cascade tiers failed. Last error: upstream timeout.",
    "type": "service_unavailable",
    "code": "cascade_exhausted"
  }
}
```

#### Examples

**Simple query (handled by draft)**

Request:
```bash
curl http://localhost:7780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Response:
```json
{
  "id": "chatcmpl-001",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "claude-haiku-3-5",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
  "cascade_info": {
    "tier": "draft",
    "confidence": 0.94,
    "variant": "psda",
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "draft_model": "claude-haiku-3-5",
    "qualifier_model": "claude-sonnet-3-5",
    "target_model": "claude-opus-4",
    "draft_ms": 298,
    "qualifier_ms": null,
    "target_ms": null,
    "total_ms": 298,
    "cost_usd": 0.000014,
    "request_id": "a1b2c3d4-0000-0000-0000-000000000001"
  }
}
```

**Complex query (escalated to target)**

Request:
```bash
curl http://localhost:7780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert in computational complexity theory."
      },
      {
        "role": "user",
        "content": "Explain the relationship between the polynomial hierarchy and PSPACE, including why most complexity theorists believe PH does not collapse to any fixed level."
      }
    ],
    "max_tokens": 2048
  }'
```

Response:
```json
{
  "id": "chatcmpl-002",
  "object": "chat.completion",
  "created": 1710000100,
  "model": "claude-opus-4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The polynomial hierarchy (PH) is a complexity-theoretic construct..."
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 72, "completion_tokens": 487, "total_tokens": 559},
  "cascade_info": {
    "tier": "target",
    "confidence": 1.0,
    "variant": "psda",
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "draft_model": "claude-haiku-3-5",
    "qualifier_model": "claude-sonnet-3-5",
    "target_model": "claude-opus-4",
    "draft_ms": 412,
    "qualifier_ms": 1830,
    "target_ms": 9240,
    "total_ms": 11482,
    "cost_usd": 0.043200,
    "request_id": "a1b2c3d4-0000-0000-0000-000000000002"
  }
}
```

---

### GET /health

Returns the health status of the proxy and its configured providers.

#### Response

**200 OK — all providers reachable**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "providers": {
    "anthropic": {"status": "ok", "latency_ms": 45},
    "openai": {"status": "ok", "latency_ms": 82},
    "openrouter": {"status": "unconfigured"}
  },
  "cascade": {
    "variant": "psda",
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "adaptive_threshold_enabled": false
  }
}
```

**200 OK — one provider degraded**

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "uptime_seconds": 7200,
  "providers": {
    "anthropic": {"status": "ok", "latency_ms": 38},
    "openai": {"status": "error", "error": "connection timeout"},
    "openrouter": {"status": "unconfigured"}
  },
  "cascade": {
    "variant": "psda",
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "adaptive_threshold_enabled": false
  }
}
```

`"status": "degraded"` is returned when at least one configured provider is unreachable. The cascade may still function if the degraded provider is not needed for the current query distribution.

**503 Service Unavailable — proxy cannot serve requests**

Returned only when the cascade cannot function at all (e.g., all providers are down and no fallback is possible).

```json
{
  "status": "error",
  "error": "No upstream providers available"
}
```

---

### GET /v1/metrics

Returns accumulated metrics for the current server process lifetime. Metrics reset on server restart.

#### Response

```json
{
  "requests": {
    "total": 15420,
    "by_tier": {
      "draft": 10794,
      "draft_fuzzy": 0,
      "qualifier": 3546,
      "target": 1080,
      "degraded": 0
    },
    "tier_fractions": {
      "draft": 0.700,
      "qualifier": 0.230,
      "target": 0.070
    }
  },
  "cost": {
    "total_usd": 12.43,
    "today_usd": 4.21,
    "by_tier_usd": {
      "draft": 0.98,
      "qualifier": 3.87,
      "target": 7.58
    },
    "estimated_savings_vs_target_only_usd": 89.12
  },
  "latency_ms": {
    "p50": 380,
    "p95": 4200,
    "p99": 11800,
    "mean": 842
  },
  "confidence": {
    "draft_mean": 0.71,
    "qualifier_mean": 0.62
  },
  "thresholds": {
    "tau_Q": 0.65,
    "tau_T": 0.80,
    "adaptive_enabled": false
  },
  "uptime_seconds": 86400
}
```

**Field Descriptions**

| Field | Description |
|-------|-------------|
| `requests.total` | Total requests processed since startup. |
| `requests.by_tier` | Count of requests answered by each tier. |
| `requests.tier_fractions` | Fraction of total answered by each tier. |
| `cost.total_usd` | Estimated total API cost since startup. |
| `cost.today_usd` | Estimated cost since midnight local time. |
| `cost.estimated_savings_vs_target_only_usd` | Estimated cost that would have been incurred if every request had been sent directly to the target model. |
| `latency_ms` | End-to-end latency percentiles over all requests. |
| `confidence.draft_mean` | Mean confidence score of draft tier responses (all requests, not just accepted ones). |
| `confidence.qualifier_mean` | Mean confidence score of qualifier tier responses (only requests that reached qualifier). |
| `thresholds` | Current tau_Q and tau_T values (may differ from config if adaptive threshold is active). |

---

### GET /v1/models

Returns the list of models configured in the cascade. Included for OpenAI SDK compatibility; clients that enumerate available models will see the cascade's tier models.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "openclaw",
      "object": "model",
      "created": 1710000000,
      "owned_by": "openclaw",
      "description": "OpenClaw 3-tier speculative cascade (psda): haiku-3-5 → sonnet-3-5 → opus-4"
    },
    {
      "id": "claude-haiku-3-5",
      "object": "model",
      "created": 1710000000,
      "owned_by": "anthropic",
      "description": "Draft tier"
    },
    {
      "id": "claude-sonnet-3-5",
      "object": "model",
      "created": 1710000000,
      "owned_by": "anthropic",
      "description": "Qualifier tier"
    },
    {
      "id": "claude-opus-4",
      "object": "model",
      "created": 1710000000,
      "owned_by": "anthropic",
      "description": "Target tier"
    }
  ]
}
```

Passing any of the listed model IDs (or `"openclaw"`, `"auto"`, or any other string) in a `/v1/chat/completions` request routes through the full cascade. The `model` field in the request body does not affect which tier handles the request.

---

## OpenAI SDK Compatibility

To use OpenClaw as a drop-in replacement for the OpenAI API via the Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7780/v1",
    api_key="not-used",  # OpenClaw does not require client auth
)

response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "user", "content": "Explain gradient descent."}
    ]
)

print(response.choices[0].message.content)

# Access cascade metadata
if hasattr(response, "cascade_info"):
    print(f"Answered by: {response.cascade_info['tier']}")
    print(f"Cost: ${response.cascade_info['cost_usd']:.6f}")
```

For streaming:

```python
with client.chat.completions.stream(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about entropy."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## cascade_info in Downstream Logging

`cascade_info` is designed to be forwarded to your application's logging and observability pipeline. A typical pattern is to log it alongside your application's own request metadata:

```python
response = client.chat.completions.create(model="auto", messages=messages)

logger.info("llm_request", extra={
    "user_id": current_user.id,
    "feature": "chat",
    "cascade_tier": response.cascade_info.get("tier"),
    "cascade_cost_usd": response.cascade_info.get("cost_usd"),
    "cascade_total_ms": response.cascade_info.get("total_ms"),
    "request_id": response.cascade_info.get("request_id"),
})
```

This enables you to correlate OpenClaw's per-request costs and tier distributions with your application's business metrics.
