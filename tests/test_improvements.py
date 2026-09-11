"""
Tests for improvements listed in improvements.md §6 (Test Coverage Gaps).

Covers:
  §2.1  CircuitBreaker concurrent probe race
  §2.2  In-flight counter invariant after any failure
  §2.3  CancelledError propagation through aclose
  §3.5  Duplicate fallback deduplication
  §4.1  Telemetry bounded memory (deque maxlen)
  §5.3  _rr_index thread safety in LatencyBasedStrategy
"""
import asyncio
import threading
import pytest
from typing import AsyncGenerator, List, Dict, Optional
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.api.v1.chat import get_providers
from app.services.llm.base import LLMProvider
from app.services.routing.circuit_breaker import CircuitBreaker, CircuitState
from app.services.routing.strategies import (
    LeastInFlightStrategy,
    LatencyBasedStrategy,
)
from app.services.routing.store import InMemoryMetricsStore
from app.services.routing.manager import RouterManager
from app.services.telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

class SimpleProvider(LLMProvider):
    """Minimal provider that yields fixed chunks or raises on first call."""

    def __init__(self, name: str, chunks: Optional[List[str]] = None, fail: bool = False):
        self.name = name
        self.chunks = chunks or [f"chunk from {name}"]
        self.fail = fail
        self.call_count = 0

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        if self.fail:
            raise RuntimeError(f"{self.name} failed intentionally")
        for c in self.chunks:
            yield c

    def get_provider_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# §2.1 — CircuitBreaker concurrent canary-probe race
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_single_canary_probe_under_concurrency():
    """
    §2.1: When circuit is HALF_OPEN, only ONE concurrent can_execute() call
    must return True (the canary probe). All others must return False.
    Without the lock, two coroutines can both see _probe_active=False and
    both proceed — the lock makes this atomic.
    """
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    # Trip the circuit
    cb.record_failure("openai")
    # recovery_timeout=0 means it transitions to HALF_OPEN immediately
    assert cb.get_state("openai") == CircuitState.HALF_OPEN

    # Run many concurrent can_execute calls
    n_concurrent = 20
    results = await asyncio.gather(
        *[asyncio.to_thread(cb.can_execute, "openai") for _ in range(n_concurrent)]
    )

    true_count = sum(1 for r in results if r)
    assert true_count == 1, (
        f"Expected exactly 1 canary probe to succeed, got {true_count} "
        f"(TOCTOU race detected — lock is not working)"
    )


# ---------------------------------------------------------------------------
# §2.2 — In-flight counter invariant after any failure type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_flight_counter_zero_after_runtime_error():
    """§2.2: In-flight counter must be 0 after RuntimeError from provider."""
    store = InMemoryMetricsStore()
    strategy = LeastInFlightStrategy(store=store)
    failing = SimpleProvider("openai", fail=True)
    manager = RouterManager([failing])

    from app.services.routing import manager as mgr_module
    orig = mgr_module.get_strategy
    mgr_module.get_strategy = lambda _: strategy

    try:
        with pytest.raises(RuntimeError):
            async for _ in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
                pass
    finally:
        mgr_module.get_strategy = orig

    assert store.get_in_flight("openai") == 0, "In-flight counter leaked after RuntimeError"


@pytest.mark.asyncio
async def test_in_flight_counter_zero_after_timeout_error():
    """§2.2: In-flight counter must be 0 after asyncio.TimeoutError from provider."""
    store = InMemoryMetricsStore()
    strategy = LeastInFlightStrategy(store=store)

    class TimeoutProvider(LLMProvider):
        async def stream_chat(self, messages, model=None, **kwargs):
            raise asyncio.TimeoutError("simulated timeout")
            yield  # make it an async generator

        def get_provider_name(self):
            return "openai"

    manager = RouterManager([TimeoutProvider()])

    from app.services.routing import manager as mgr_module
    orig = mgr_module.get_strategy
    mgr_module.get_strategy = lambda _: strategy

    try:
        with pytest.raises((asyncio.TimeoutError, Exception)):
            async for _ in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
                pass
    finally:
        mgr_module.get_strategy = orig

    assert store.get_in_flight("openai") == 0, "In-flight counter leaked after TimeoutError"


@pytest.mark.asyncio
async def test_in_flight_counter_zero_after_stop_async_iteration():
    """§2.2: In-flight counter must be 0 when provider yields no chunks (StopAsyncIteration)."""
    store = InMemoryMetricsStore()
    strategy = LeastInFlightStrategy(store=store)

    class EmptyProvider(LLMProvider):
        async def stream_chat(self, messages, model=None, **kwargs):
            return
            yield  # async generator that immediately stops

        def get_provider_name(self):
            return "openai"

    manager = RouterManager([EmptyProvider()])

    from app.services.routing import manager as mgr_module
    orig = mgr_module.get_strategy
    mgr_module.get_strategy = lambda _: strategy

    try:
        async for _ in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
            pass
    finally:
        mgr_module.get_strategy = orig

    assert store.get_in_flight("openai") == 0, "In-flight counter leaked after empty stream"


# ---------------------------------------------------------------------------
# §2.3 — CancelledError propagation through aclose
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancelled_error_propagates_through_aclose():
    """
    §2.3: When a streaming task is cancelled mid-flight, the CancelledError
    must propagate out of _stream_with_timeouts. Before the fix, a bare
    'except Exception: pass' in the finally block could swallow it.
    """
    class InfiniteProvider(LLMProvider):
        async def stream_chat(self, messages, model=None, **kwargs):
            for i in range(10_000):
                await asyncio.sleep(0)  # yield control so cancel can land
                yield f"chunk{i}"

        def get_provider_name(self):
            return "openai"

    manager = RouterManager([InfiniteProvider()])

    async def consume():
        async for _ in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
            pass

    task = asyncio.create_task(consume())
    # Let it start
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.done() and task.cancelled(), "Task was not fully cancelled"


# ---------------------------------------------------------------------------
# §3.5 — Duplicate fallback deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_fallback_models_attempted_only_once():
    """
    §3.5: When fallback_models contains the same model twice, the provider
    should be attempted only once (deduplication by (provider_name, model) key).
    """
    provider = SimpleProvider("openai", fail=True)

    with pytest.raises(RuntimeError):
        async for _ in RouterManager([provider]).stream_with_fallback(
            [{"role": "user", "content": "hi"}],
            preference="gpt-4o",
            fallback_models=["gpt-4o", "gpt-4o"],  # duplicate
        ):
            pass

    # Without dedup the provider would be called 3× (primary + 2 fallbacks).
    # With dedup it should be called exactly once since all entries resolve to
    # the same (openai, gpt-4o) pair.
    assert provider.call_count == 1, (
        f"Expected provider called once after dedup, got {provider.call_count}"
    )


# ---------------------------------------------------------------------------
# §4.1 — Telemetry bounded memory
# ---------------------------------------------------------------------------

def test_telemetry_ttfc_bounded_at_maxlen():
    """§4.1: Inserting > 10,000 TTFC samples must not grow the deque beyond maxlen."""
    collector = TelemetryCollector()
    for i in range(15_000):
        collector.record_ttfc(float(i))
    assert len(collector.ttfc_latencies_ms) == 10_000, (
        f"Expected deque length 10000, got {len(collector.ttfc_latencies_ms)}"
    )


def test_telemetry_duration_bounded_at_maxlen():
    """§4.1: Inserting > 10,000 duration samples must not grow the deque beyond maxlen."""
    collector = TelemetryCollector()
    for i in range(12_000):
        collector.record_request_complete(float(i), success=True)
    assert len(collector.total_durations_ms) == 10_000, (
        f"Expected deque length 10000, got {len(collector.total_durations_ms)}"
    )


def test_telemetry_get_metrics_correct_after_overflow():
    """§4.1 + §5.2: get_metrics() must return consistent percentiles after deque overflow."""
    collector = TelemetryCollector()
    # Fill with 12k samples — the oldest 2k will be evicted
    for i in range(12_000):
        collector.record_ttfc(float(i))
    metrics = collector.get_metrics()
    assert metrics["ttfc_ms"]["count"] == 10_000
    assert metrics["ttfc_ms"]["p50"] is not None
    assert metrics["ttfc_ms"]["p99"] is not None


# ---------------------------------------------------------------------------
# §5.3 — _rr_index thread-safety in LatencyBasedStrategy
# ---------------------------------------------------------------------------

def test_rr_index_thread_safe_explores_all_unknown_providers():
    """
    §5.3: Under concurrent load, round-robin across unknown providers must
    ensure that every unknown provider is eventually selected (not just one).
    The _rr_lock prevents two threads from both reading the same index.
    """
    store = InMemoryMetricsStore()
    strategy = LatencyBasedStrategy(store=store)

    p_a = SimpleProvider("providerA")
    p_b = SimpleProvider("providerB")
    providers = [p_a, p_b]

    # Neither provider has latency data → both are "unknown"
    selected_names = []
    lock = threading.Lock()

    def select():
        result = strategy.select_provider(providers)
        with lock:
            selected_names.append(result.get_provider_name())

    threads = [threading.Thread(target=select) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique = set(selected_names)
    assert "providerA" in unique, "providerA was never selected (round-robin broken)"
    assert "providerB" in unique, "providerB was never selected (round-robin broken)"


# ---------------------------------------------------------------------------
# §4.2 — /v1/health/routing endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_health_endpoint_returns_provider_info():
    """§4.2: GET /v1/health/routing must return per-provider health data."""
    p1 = SimpleProvider("openai", ["hello"])
    app.dependency_overrides[get_providers] = lambda: [p1]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/v1/health/routing")
        assert response.status_code == 200
        body = response.json()
        assert "providers" in body
        assert "openai" in body["providers"]
        prov = body["providers"]["openai"]
        assert "circuit_state" in prov
        assert "in_flight" in prov
        assert "ema_error_rate" in prov
        assert "fallback_events" in body
    finally:
        app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# §4.3 — Structured fallback events
# ---------------------------------------------------------------------------

def test_structured_fallback_events_recorded():
    """§4.3: record_fallback() must store structured events accessible via get_fallback_events()."""
    collector = TelemetryCollector()
    collector.record_fallback(
        failed_provider="openai",
        reason="asyncio.TimeoutError: TTFC timeout",
        next_provider="anthropic",
    )
    events = collector.get_fallback_events()
    assert len(events) == 1
    assert events[0]["failed_provider"] == "openai"
    assert events[0]["next_provider"] == "anthropic"
    assert "timeout" in events[0]["reason"].lower()
    assert "timestamp" in events[0]


def test_fallback_events_ring_buffer_bounded():
    """§4.3: Fallback events ring-buffer must be bounded at maxlen=200."""
    collector = TelemetryCollector()
    for i in range(300):
        collector.record_fallback(
            failed_provider=f"p{i}",
            reason="error",
            next_provider=f"p{i+1}",
        )
    events = collector.get_fallback_events()
    assert len(events) == 200, f"Expected 200, got {len(events)}"
    # Most recent events should be present
    assert events[-1]["failed_provider"] == "p299"


# ---------------------------------------------------------------------------
# §2.5 — No providers configured → HTTP 503
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_providers_returns_503():
    """§2.5: When get_providers raises ValueError (no keys configured), endpoint returns 503."""
    def no_providers():
        raise ValueError("No LLM provider API keys are configured. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY.")

    app.dependency_overrides[get_providers] = no_providers

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 503, f"Expected 503, got {response.status_code}"
    finally:
        app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# §3.6 — Model pricing normalization
# ---------------------------------------------------------------------------

def test_get_model_cost_friendly_name_resolves_consistently():
    """§3.6: get_model_cost with friendly name and actual ID must return same cost."""
    from app.core.models import get_model_cost
    friendly_cost = get_model_cost("anthropic", "claude-3-haiku")
    actual_cost = get_model_cost("anthropic", "claude-3-haiku-20240307")
    assert friendly_cost == actual_cost, (
        f"Friendly name cost ({friendly_cost}) != actual model ID cost ({actual_cost})"
    )


def test_get_model_cost_2026_friendly_names_resolve():
    """§3.6: 2026 friendly names resolve correctly through MODEL_MAPPING."""
    from app.core.models import get_model_cost
    # gpt-5.4-mini -> gpt-4o-mini
    mini_cost = get_model_cost("openai", "gpt-5.4-mini")
    direct_cost = get_model_cost("openai", "gpt-4o-mini")
    assert mini_cost == direct_cost


# ---------------------------------------------------------------------------
# §3.3 — Strategy registry replaces lru_cache (invalidation in tests)
# ---------------------------------------------------------------------------

def test_strategy_registry_returns_correct_types():
    """§3.3: get_strategy must return the correct singleton per name."""
    from app.services.routing.manager import get_strategy, _STRATEGY_REGISTRY
    from app.services.routing.strategies import (
        HardcodedStrategy, LeastInFlightStrategy,
        LatencyBasedStrategy, CostLatencyTradeoffStrategy,
    )

    assert isinstance(get_strategy("hardcoded"), HardcodedStrategy)
    assert isinstance(get_strategy("load_balance"), LeastInFlightStrategy)
    assert isinstance(get_strategy("latency"), LatencyBasedStrategy)
    assert isinstance(get_strategy("cost_latency"), CostLatencyTradeoffStrategy)
    # Unknown names fall back to hardcoded
    assert isinstance(get_strategy("does_not_exist"), HardcodedStrategy)


def test_strategy_registry_can_be_overridden_in_tests():
    """§3.3: The dict registry allows test-time injection without monkeypatching lru_cache."""
    from app.services.routing.manager import _STRATEGY_REGISTRY, get_strategy
    from app.services.routing.strategies import RoutingStrategy, HardcodedStrategy

    class DummyStrategy(RoutingStrategy):
        def select_provider(self, providers, preference=None):
            return providers[0]

    original = _STRATEGY_REGISTRY.get("test_strategy")
    try:
        _STRATEGY_REGISTRY["test_strategy"] = DummyStrategy()
        result = get_strategy("test_strategy")
        assert isinstance(result, DummyStrategy)
    finally:
        if original is None:
            _STRATEGY_REGISTRY.pop("test_strategy", None)
        else:
            _STRATEGY_REGISTRY["test_strategy"] = original

