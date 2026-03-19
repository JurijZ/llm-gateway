




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
