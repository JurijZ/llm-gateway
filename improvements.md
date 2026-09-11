# LLM Gateway — Routing Service Improvements Plan

> Reviewed: `app/services/routing/`, `app/services/llm/`, `app/core/`, `app/api/v1/chat.py`  
> Date: 2026-09-12

---

## 1. Architecture Overview

```
ChatRequest → chat.py (API) → RouterManager.stream_with_fallback()
                                   │
                    ┌──────────────┼──────────────────────┐
                    ▼              ▼                       ▼
             _build_candidates  get_strategy()     CircuitBreaker
             (ordered fallback   (lru_cache,        (per-provider
              chain)              singleton)         open/half-open)
                    │
                    ▼
             _stream_with_timeouts(provider)
               Phase 1: TTFC_TIMEOUT (10 s)
               Phase 2: CHUNK_TIMEOUT idle (30 s)
                    │
                    ▼
             strategy hooks (on_request_start / on_first_chunk / on_request_success / on_request_error / on_request_end)
                    │
                    ▼
             MetricsStore (InMemory or Redis)
             TelemetryCollector (in-memory)
```

---

## 2. Priority 1 — Correctness & Reliability

### 2.1 Thread/async-safety of `CircuitBreaker`

**File:** `circuit_breaker.py`

**Problem:** The `CircuitBreaker` state machine uses plain `dict` fields with no locks.
`get_state()` reads `_states` and then _writes_ it (OPEN → HALF_OPEN transition) in a
non-atomic way. In an async environment with concurrent coroutines the TOCTOU window
can cause two simultaneous probes to both see HALF_OPEN and both mark `_probe_active = True`,
defeating the single-canary-probe guarantee.

**Improvement:**
- Add an `asyncio.Lock` (or `threading.Lock` for thread-safety with sync callers) around
  the `get_state` / `can_execute` / `record_success` / `record_failure` mutating paths.
- Alternatively, collapse `get_state()` + `can_execute()` into one atomic method that
  evaluates and advances state in a single critical section.

```python
# circuit_breaker.py
import threading

class CircuitBreaker:
    def __init__(self, ...):
        ...
        self._lock = threading.Lock()

    def can_execute(self, provider_name: str) -> bool:
        with self._lock:
            state = self._get_state_unlocked(provider_name)
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN and not self._probe_active.get(provider_name, False):
                self._probe_active[provider_name] = True
                return True
            return False
```

---

### 2.2 `LeastInFlightStrategy` — decrement invariant is fragile

**File:** `manager.py` lines 260–310, `strategies.py` lines 67–71

**Problem:** `on_request_start` increments in-flight, `on_request_end` decrements (in
`finally`). The `finally` block at line 308 does call `on_request_end`, so this path is
technically correct — but the try/except/finally block ordering is fragile and non-obvious.
A misread or future refactor could easily break it.

**Improvement:**
- Document the invariant explicitly: every `on_request_start` MUST be paired with
  `on_request_end` regardless of exception type.
- Add a unit test that verifies the in-flight counter is always zero after any failure
  type (timeout, RuntimeError, StopAsyncIteration).

---

### 2.3 `_stream_with_timeouts` swallows `CancelledError` in `aclose()`

**File:** `manager.py` lines 192–197

**Problem:** The `aiter.aclose()` call in the `_stream_with_timeouts` finally block silently
swallows _any_ exception from close (`except Exception: pass`). If `aclose` raises
`asyncio.CancelledError` (a `BaseException` subclass in Python 3.8+), it escapes the bare
`except Exception` — but earlier Python patterns and some ASGI servers route cancellation as
an `Exception`. Worse, `GeneratorExit` during generator cleanup is also silently lost.

**Improvement:**
```python
# manager.py _stream_with_timeouts finally block
finally:
    try:
        await aiter.aclose()
    except (GeneratorExit, asyncio.CancelledError):
        raise   # let cancellation propagate
    except Exception:
        pass
```

---

### 2.4 `select_provider` may return `None` in `CostLatencyTradeoffStrategy`

**File:** `strategies.py` lines 197–199

**Problem:** `select_provider` returns `None` when `providers` is empty. The return
type annotation says `-> LLMProvider`, not `-> Optional[LLMProvider]`. The `RouterManager`
never passes an empty list today, but this is a latent type violation that will cause a
`NoneType` attribute error on `provider.get_provider_name()` at runtime.

**Improvement:**
- Change return type to `Optional[LLMProvider]` across the `RoutingStrategy` ABC, OR
- Raise `ValueError("No providers available")` when `providers` is empty (preferred — fail-fast).
- Add a guard in `RouterManager._build_candidates` asserting `self.providers` is non-empty.

---

### 2.5 Hardcoded fallback to unconfigured providers

**File:** `chat.py` lines 38–42

**Problem:** When neither API key is set, `get_providers()` silently adds both
`OpenAIProvider()` and `AnthropicProvider()` with `key=None`. Every actual request
will fail at the SDK level with an auth error, but only after the full TTFC timeout
(10 s × 2 providers = 20 s). This wastes time and generates misleading logs.

**Improvement:**
- Raise a startup-time `ValueError` or log a `CRITICAL` warning if no providers are
  configured, instead of returning dummy unauthenticated providers.
- OR: return `503 Service Unavailable` immediately when all provider keys are missing.

---

## 3. Priority 2 — Design & Maintainability

### 3.1 Proxy objects in `strategies.py` are over-engineered

**File:** `strategies.py` lines 46–99, 131–172

**Problem:** `LeastInFlightStrategy.in_flight`, `LatencyBasedStrategy.latencies`, and
`CostLatencyTradeoffStrategy.latencies/costs/error_rates` return nested proxy classes that
simulate dict semantics by delegating to `MetricsStore`. These proxies:
- Are only used in tests (they exist to support `strategy.latencies["openai"] = 0.5`).
- Introduce hidden coupling to `InMemoryMetricsStore` internals (`hasattr(self._store, "_latencies")`).
- Violate the `MetricsStore` abstraction by reaching into private fields.

**Improvement:**
- Remove the proxy properties entirely; expose `update_*` / `get_*` methods directly on strategies.
- In tests, call `strategy.update_latency("openai", 0.5)` instead of `strategy.latencies["openai"] = 0.5`.
- This removes ~80 lines of proxy boilerplate.

---

### 3.2 Duplicate latency-update logic between strategies

**File:** `strategies.py` lines 101–116 and 174–189

**Problem:** `LatencyBasedStrategy` and `CostLatencyTradeoffStrategy` both define an
inner `LatenciesProxy` class with identical code and duplicate the EMA update delegation
(`on_first_chunk` → `store.update_latency`).

**Improvement:**
- Extract a `_LatencyAwareStrategy(RoutingStrategy)` base class that provides the shared
  `update_latency` / `on_first_chunk` implementation.
- Both concrete strategies inherit from it and only override `select_provider`.

---

### 3.3 `get_strategy` uses `lru_cache` — hard to test and extend

**File:** `manager.py` lines 17–26

**Problem:** `@lru_cache(maxsize=None)` ensures strategies are singletons, which is correct
for preserving metrics state. However:
- There is no way to invalidate the cache in tests without monkeypatching the module.
- Adding a new strategy requires modifying the function body (closed to extension).

**Improvement:**
Replace with an explicit registry singleton:

```python
# manager.py
_STRATEGY_REGISTRY: dict[str, RoutingStrategy] = {}

def _init_registry() -> None:
    store = get_metrics_store()
    _STRATEGY_REGISTRY.update({
        "hardcoded":    HardcodedStrategy(),
        "load_balance": LeastInFlightStrategy(store),
        "latency":      LatencyBasedStrategy(store),
        "cost_latency": CostLatencyTradeoffStrategy(store=store),
    })

def get_strategy(name: str) -> RoutingStrategy:
    if not _STRATEGY_REGISTRY:
        _init_registry()
    return _STRATEGY_REGISTRY.get(name, _STRATEGY_REGISTRY["hardcoded"])
```

---

### 3.4 `RouterManager` mutable result fields (`last_selected_provider/model`)

**File:** `chat.py` lines 44–47, `manager.py` lines 42–43

**Problem:** `RouterManager` stores `last_selected_provider` and `last_selected_model` as
mutable instance fields set mid-stream. While each request gets its own manager instance
(no race condition), the pattern is fragile: result metadata is encoded as side-effect state
rather than as a return value, making unit testing harder.

**Improvement:**
- Return `(async_generator, metadata_future)` from `stream_with_fallback` where
  `metadata_future` is an `asyncio.Future` resolved on first-chunk commit, OR
- Pass a `result: dict` out-parameter that the caller can inspect after the first chunk.

---

### 3.5 Duplicate entries possible in fallback chain

**File:** `manager.py` lines 114–126

**Problem:** `fallback_models` can list the _same model_ twice. Both entries will be added
to candidates, causing redundant retry attempts against the same provider+model pair.

**Improvement:**
Deduplicate candidates after building the chain:

```python
seen: set[tuple[str, str | None]] = set()
deduped: list[tuple[LLMProvider, str | None]] = []
for p, m in candidates:
    key = (p.get_provider_name(), m)
    if key not in seen:
        seen.add(key)
        deduped.append((p, m))
candidates = deduped
```

---

### 3.6 Model pricing inconsistency with model name resolution

**File:** `core/models.py`

**Problem:**
- `MODEL_MAPPING` maps friendly names → `(provider, actual_model_id)`.
- `MODEL_PRICING` maps a mix of friendly names AND actual model IDs to costs.
- `get_model_cost(provider, model)` looks up `model` directly in `MODEL_PRICING`.
  Whether it finds a friendly-name entry or an actual-model-ID entry depends on
  which layer resolved the name, creating inconsistent cost values for the same model.
- Several entries are duplicated (e.g. `"claude-3-opus"` and `"claude-3-opus-20240229"`).

**Improvement:**
- Normalize: always resolve friendly name → actual model ID first, then look up cost
  by actual model ID only. Remove friendly-name entries from `MODEL_PRICING`.
- Consider loading pricing from a config file (YAML/JSON) to allow updates without code changes.

---

## 4. Priority 3 — Observability & Operability

### 4.1 `TelemetryCollector` has unbounded memory growth

**File:** `telemetry.py` lines 18–19

**Problem:** `ttfc_latencies_ms` and `total_durations_ms` are plain lists that grow without
bound. Under production load these lists consume significant memory and cause O(n log n) sort
cost on every `get_metrics()` call (p50/p95/p99 are computed by sorting the full list).

**Improvement:**
```python
from collections import deque

self.ttfc_latencies_ms: deque[float] = deque(maxlen=10_000)
self.total_durations_ms: deque[float] = deque(maxlen=10_000)
```

For percentile accuracy under a bounded window, T-Digest or reservoir sampling can be added.

---

### 4.2 No routing-health introspection endpoint

**Problem:** `GET /health` returns only `{"status": "ok"}`. Operators cannot inspect:
- Current circuit breaker state per provider.
- Current in-flight request counts.
- EMA latency and error rates per provider.

**Improvement:**
Expose `GET /v1/health/routing` returning:

```json
{
  "providers": {
    "openai":    { "circuit_state": "CLOSED",    "in_flight": 3, "ema_latency_ms": 412.5, "ema_error_rate": 0.02 },
    "anthropic": { "circuit_state": "HALF_OPEN",  "in_flight": 0, "ema_latency_ms": 650.1, "ema_error_rate": 0.18 }
  }
}
```

---

### 4.3 Fallback events lack structured context

**File:** `manager.py` line 257

**Problem:** `get_telemetry().record_fallback()` increments a bare counter. It does not
record which provider failed, the failure reason, or which provider was tried next. This
makes debugging fallback storms in production impossible.

**Improvement:**
- Record structured fallback events: `(timestamp, request_id, failed_provider, reason, next_provider)`.
- Surface as a ring-buffer of recent events in the `/v1/telemetry` endpoint.

---

### 4.4 Strategy selection not logged as structured fields

**File:** `manager.py` lines 238–241

**Problem:** Candidate list is logged as an f-string. With `JSON_LOGS=True`, this is
embedded in a string field and cannot be queried or alerted on per-provider.

**Improvement:**
```python
logger.info(
    "Routing decision",
    extra={
        "strategy": type(active_strategy).__name__,
        "candidates": [{"provider": p.get_provider_name(), "model": m} for p, m in candidates],
    }
)
```

---

## 5. Priority 4 — Performance

### 5.1 `RedisMetricsStore.update_latency` is not atomic

**File:** `store.py` lines 179–188

**Problem:** `update_latency` performs GET then SET in two separate Redis commands.
Under concurrent load from multiple workers, two workers can both read the same stale value,
compute independent EMAs, and one will overwrite the other.

**Improvement:**
Use a Lua script to perform the EMA update atomically:

```lua
-- latency_ema.lua
local cur = redis.call('GET', KEYS[1])
local new_val
if cur == false then
    new_val = tonumber(ARGV[1])
else
    new_val = tonumber(cur) * tonumber(ARGV[2]) + tonumber(ARGV[1]) * (1 - tonumber(ARGV[2]))
end
redis.call('SET', KEYS[1], new_val)
return tostring(new_val)
```

Apply the same pattern to `update_error_rate`.

---

### 5.2 `_calc_percentile` sorts the full list on every call

**File:** `telemetry.py` lines 52–63, 65–106

**Problem:** `get_metrics()` calls `_calc_percentile` three times for TTFC and three times
for duration. Each call re-sorts the list. All six sorts happen while holding `_lock`,
blocking concurrent telemetry writes.

**Improvement:**
Sort once per `get_metrics()` call and reuse:

```python
def get_metrics(self) -> Dict:
    with self._lock:
        sorted_ttfc = sorted(self.ttfc_latencies_ms)
        ttfc_p50 = self._calc_percentile(sorted_ttfc, 0.50)
        ttfc_p95 = self._calc_percentile(sorted_ttfc, 0.95)
        ttfc_p99 = self._calc_percentile(sorted_ttfc, 0.99)
        ...
```

---

### 5.3 `LatencyBasedStrategy._rr_index` is not thread-safe

**File:** `strategies.py` lines 111–114

**Problem:** The round-robin index for providers with no latency history is an unprotected
integer. In a threaded ASGI server, two concurrent requests can read the same index and
both select the same "unknown" provider, skipping latency exploration of other providers.

**Improvement:**
```python
import threading

class LatencyBasedStrategy(RoutingStrategy):
    def __init__(self, ...):
        ...
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def select_provider(self, providers, preference=None):
        ...
        if unknown:
            with self._rr_lock:
                idx = self._rr_index % len(unknown)
                self._rr_index += 1
            return unknown[idx]
```

---

## 6. Test Coverage Gaps

| Gap | Suggested Test |
|-----|---------------|
| CircuitBreaker concurrent probe race (§2.1) | `asyncio.gather` two `can_execute` calls on HALF_OPEN circuit; assert only one returns `True` |
| In-flight counter invariant after any failure (§2.2) | Verify counter is 0 after TimeoutError, RuntimeError, and StopAsyncIteration |
| `CancelledError` propagation through `aclose` (§2.3) | Cancel streaming task mid-flight; verify task fully cancelled |
| Duplicate fallback deduplication (§3.5) | Pass `fallback_models=["gpt-4o", "gpt-4o"]`; verify provider attempted only once |
| `RedisMetricsStore` atomic EMA (§5.1) | N concurrent writers updating latency; assert final EMA within tolerance |
| Telemetry bounded memory (§4.1) | Insert >10,000 samples; assert list length stays bounded |
| `_rr_index` thread safety (§5.3) | Concurrent `select_provider` calls; assert both unknown providers are explored |

---

## 7. Summary Table

| # | Area | Severity | Effort |
|---|------|----------|--------|
| 2.1 | CircuitBreaker race condition | 🔴 High | Small |
| 2.2 | In-flight decrement invariant documentation | 🟡 Medium | Tiny |
| 2.3 | `CancelledError` swallowed in `aclose` | 🔴 High | Tiny |
| 2.4 | `None` return from `select_provider` | 🟡 Medium | Small |
| 2.5 | Unconfigured providers silently added | 🟡 Medium | Small |
| 3.1 | Proxy objects over-engineering | 🟢 Low | Medium |
| 3.2 | Duplicate latency EMA logic | 🟢 Low | Small |
| 3.3 | `lru_cache` strategy registry | 🟢 Low | Small |
| 3.4 | Mutable result fields on `RouterManager` | 🟢 Low | Medium |
| 3.5 | Duplicate fallback entries | 🟡 Medium | Tiny |
| 3.6 | Inconsistent model pricing lookup | 🟡 Medium | Small |
| 4.1 | Unbounded telemetry lists | 🔴 High | Small |
| 4.2 | No routing health endpoint | 🟡 Medium | Medium |
| 4.3 | Fallback events unstructured | 🟡 Medium | Small |
| 4.4 | Strategy selection not structured logging | 🟢 Low | Tiny |
| 5.1 | Redis EMA update not atomic | 🔴 High | Medium |
| 5.2 | Redundant sort in `get_metrics` | 🟢 Low | Tiny |
| 5.3 | `_rr_index` not thread-safe | 🟡 Medium | Tiny |
