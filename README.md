
## Start coding agents
```bash
npx @google/gemini-cli
```

### Start FastAPI server
```bash
cd app/
uv run fastapi dev main.py
```

Swagger is on http://127.0.0.1:8000/docs

---

## Routing Strategies

The gateway supports four routing strategies that control which LLM provider handles each request. The strategy is selected per-request via the `routing_strategy` field, or set globally via the `DEFAULT_STRATEGY` environment variable.

### Available strategies

| Strategy ID | Name | Description |
|---|---|---|
| `hardcoded` | Hardcoded | Returns the first configured provider, or the one matching `model_preference`. Default strategy. |
| `load_balance` | Least In-Flight | Routes to the provider with the fewest active requests, spreading load evenly. |
| `latency` | Latency-Based | Tracks rolling-average latency (time to first chunk) per provider and always routes to the fastest one. Unknown providers are round-robined to gather data first. |
| `cost_latency` | Cost + Latency Tradeoff | Scores each provider with `α×(1/latency) + β×(1/cost) + γ×(1−error_rate)` and picks the highest score. Weights default to α=0.4, β=0.4, γ=0.2. Providers with a 100% error rate are excluded when alternatives exist. |

### Strategy vs. model preference

When a `routing_strategy` is set, **the strategy controls which provider is used** — `model_preference` no longer forces a specific provider. The model hint is still used to select the right model *if* the strategy happens to pick the matching provider; otherwise the provider's default model is used.

For the default `hardcoded` strategy, `model_preference` continues to determine both the provider and model.

### API usage

**Use a specific model (hardcoded strategy, model forces the provider):**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "model_preference": "gpt-5.2",
  "stream": true
}
```

**Use a routing strategy (strategy selects the provider):**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "routing_strategy": "load_balance",
  "stream": true
}
```

**Combine a strategy with a model hint (strategy picks provider, model used if provider matches):**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "routing_strategy": "latency",
  "model_preference": "claude-3-5-sonnet",
  "stream": true
}
```

**No preference — falls back to the configured default strategy:**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "stream": true
}
```

### Supported model names

| Model name | Provider | Model ID |
|---|---|---|
| `gpt-4o` | openai | `gpt-4o` |
| `gpt-5.2` | openai | `gpt-5.2` |
| `gpt-3.5-turbo` | openai | `gpt-3.5-turbo` |
| `claude-3-5-sonnet` | anthropic | `claude-3-5-sonnet-20240620` |
| `claude-3-opus` | anthropic | `claude-3-opus-20240229` |
| `claude-3-haiku` | anthropic | `claude-3-haiku-20240307` |

You can also pass a provider name directly (`openai`, `anthropic`) to use that provider's default model.

### Listing strategies via the API

```bash
GET /v1/strategies
```

---

## Fallback Strategies

Fallback is the mechanism that automatically retries a request against an alternative model or provider when the primary attempt fails. It is configured independently of the routing strategy — the two compose cleanly.

### How it works

Every request is resolved into an ordered **candidate list** before any network call is made:

```
[strategy-selected primary]  →  [fallback_models in order]  →  [remaining providers]
```

The gateway works through the list left-to-right. As soon as one candidate succeeds the response is returned. If all candidates fail, the last error is surfaced to the client.

The routing strategy is only involved in selecting the **primary** candidate. The fallback chain is purely sequential and model-driven.

### The commit boundary

Because responses are streamed, fallback has a hard constraint: **it can only trigger before the first chunk is sent to the client**.

| State | Failure behaviour |
|---|---|
| Before first chunk | Try the next candidate transparently. The client sees no error. |
| After first chunk | Error is raised immediately. The partial response has already been written to the HTTP connection — there is nothing to retract. |

This means fallback protects against providers that are down, rate-limited, or too slow to respond (TTFC timeout), but not against providers that start streaming and then stall or crash mid-response.

### Timeout settings

Two independent timeouts control each individual streaming attempt:

| `.env` key | Default | Triggers fallback? | Meaning |
|---|---|---|---|
| `TTFC_TIMEOUT` | `10` s | Yes | Max time to wait for the **first chunk**. Exceeding this counts as a pre-commit failure; the next candidate is tried. |
| `CHUNK_TIMEOUT` | `30` s | No | Max idle time **between consecutive chunks**. Exceeding this means the stream stalled after committing; the error is propagated to the client. |

### Candidate list construction

Given a request, the candidate list is built in three steps:

1. **Primary** — the provider selected by the active routing strategy (using `model_preference` to resolve a specific model for that provider, if applicable).
2. **Explicit fallbacks** — each entry in `fallback_models`, resolved in declared order. The same provider may appear more than once with different models (e.g. `gpt-4o` then `gpt-3.5-turbo` are both OpenAI).
3. **Last-resort** — any configured provider not yet present in the list is appended at its default model. This ensures there is always a final safety net even if `fallback_models` is not set.

### API usage

**Explicit cross-provider fallback — try gpt-4o, then claude-3-5-sonnet, then gpt-3.5-turbo:**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "model_preference": "gpt-4o",
  "fallback_models": ["claude-3-5-sonnet", "gpt-3.5-turbo"],
  "stream": true
}
```

**Same-provider model downgrade — prefer gpt-4o, fall back to cheaper gpt-3.5-turbo:**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "model_preference": "gpt-4o",
  "fallback_models": ["gpt-3.5-turbo"],
  "stream": true
}
```

**Routing strategy with explicit fallback — cost_latency picks the primary, haiku is the safety net:**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "routing_strategy": "cost_latency",
  "fallback_models": ["claude-3-haiku"],
  "stream": true
}
```

**No fallback_models — remaining providers are still tried as a last resort:**
```json
{
  "messages": [
    { "role": "user", "content": "How many planets in the solar system?" }
  ],
  "model_preference": "gpt-4o",
  "stream": true
}
```
If OpenAI is unavailable, the gateway will automatically try the Anthropic provider at its default model before returning an error.

### Combining routing strategies and fallback

The two features operate at different levels and compose without conflict:

| Concern | Controlled by |
|---|---|
| Which provider to try **first** | Routing strategy (`routing_strategy`) |
| Which model to use on the primary provider | `model_preference` |
| What to try **if the primary fails** | `fallback_models` + last-resort providers |
| How long to wait before giving up on a provider | `TTFC_TIMEOUT` / `CHUNK_TIMEOUT` in `.env` |

A typical production configuration might use `cost_latency` to optimise the happy path while `fallback_models` guarantees availability even when the winning provider is down.
