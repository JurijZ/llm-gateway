import pytest
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator, List, Dict, Optional
import asyncio
import logging
import json
from unittest.mock import MagicMock
from pydantic import SecretStr, ValidationError

from app.main import app, lifespan
from app.core.config import Settings
from app.core.logging import StructuredJsonFormatter, StandardCorrelationFormatter, setup_logging
from app.core.middleware import request_id_ctx_var
from app.models.schemas import ChatRequest, Message
from app.api.v1.chat import get_providers
from app.services.llm.base import LLMProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.anthropic import AnthropicProvider
from app.services.routing.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    get_circuit_breaker,
)
from app.services.routing.store import InMemoryMetricsStore, RedisMetricsStore
from app.services.routing.strategies import LeastInFlightStrategy, LatencyBasedStrategy
from app.services.routing.manager import RouterManager
from app.services.telemetry import get_telemetry

class MockProvider(LLMProvider):
    def __init__(self, name: str, fail: bool = False, delay: float = 0.0):
        self.name = name
        self.fail = fail
        self.delay = delay
        self.closed = False
        self.last_temperature = None
        self.last_max_tokens = None
        self.last_top_p = None

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        self.last_top_p = top_p
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        yield f"Hello from {self.name}"

    def get_provider_name(self) -> str:
        return self.name

    async def close(self) -> None:
        self.closed = True


# ----------------------------------------------------------------------
# 1. Proposal 8: Configuration & Validation Hardening
# ----------------------------------------------------------------------

def test_secret_str_config():
    """Verify settings parses SecretStr and hides plain values."""
    s = Settings(OPENAI_API_KEY="sk-test-secret-123")
    assert isinstance(s.OPENAI_API_KEY, SecretStr)
    assert s.OPENAI_API_KEY.get_secret_value() == "sk-test-secret-123"
    assert "sk-test-secret-123" not in repr(s.OPENAI_API_KEY)


def test_schema_role_and_content_validation():
    """Verify Message role is strictly validated to system/user/assistant."""
    # Valid
    m1 = Message(role="user", content="hello")
    m2 = Message(role="assistant", content="hi there")
    m3 = Message(role="system", content="you are helpful")
    assert m1.role == "user"

    # Invalid role
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="hello")

    # Empty content
    with pytest.raises(ValidationError):
        Message(role="user", content="")

    # Empty messages list in ChatRequest
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_schema_hyperparameters_validation():
    """Verify temperature and max_tokens bounds."""
    m = Message(role="user", content="hello")
    # Valid
    req = ChatRequest(messages=[m], temperature=0.7, max_tokens=1000)
    assert req.temperature == 0.7
    assert req.max_tokens == 1000

    # Temperature > 2.0
    with pytest.raises(ValidationError):
        ChatRequest(messages=[m], temperature=2.5)

    # Temperature < 0.0
    with pytest.raises(ValidationError):
        ChatRequest(messages=[m], temperature=-0.1)

    # Max tokens < 1
    with pytest.raises(ValidationError):
        ChatRequest(messages=[m], max_tokens=0)

    # Valid top_p
    req_top_p = ChatRequest(messages=[m], top_p=0.9)
    assert req_top_p.top_p == 0.9

    # top_p > 1.0
    with pytest.raises(ValidationError):
        ChatRequest(messages=[m], top_p=1.5)

    # top_p < 0.0
    with pytest.raises(ValidationError):
        ChatRequest(messages=[m], top_p=-0.1)


@pytest.mark.asyncio
async def test_top_p_propagation():
    """Verify top_p parameter is passed from request through manager to provider."""
    p = MockProvider("openai")
    app.dependency_overrides[get_providers] = lambda: [p]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                "/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.8,
                    "max_tokens": 500,
                    "top_p": 0.95,
                    "stream": False,
                },
            )
            assert res.status_code == 200
            assert p.last_temperature == 0.8
            assert p.last_max_tokens == 500
            assert p.last_top_p == 0.95
    finally:
        app.dependency_overrides = {}


# ----------------------------------------------------------------------
# 2. Proposal 7: Correlation ID & Observability Headers
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlation_id_propagation():
    """Verify X-Request-ID header is propagated or generated."""
    p = MockProvider("openai")
    app.dependency_overrides[get_providers] = lambda: [p]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Case A: Client provides X-Request-ID
            custom_id = "custom-client-trace-id-999"
            res = await ac.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
                headers={"X-Request-ID": custom_id},
            )
            assert res.status_code == 200
            assert res.headers.get("x-request-id") == custom_id
            assert "x-ttfc-ms" in res.headers
            assert "x-total-duration-ms" in res.headers

            # Case B: Gateway generates UUID when header is omitted
            res2 = await ac.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert res2.status_code == 200
            assert res2.headers.get("x-request-id") is not None
            assert len(res2.headers.get("x-request-id")) > 10
            assert "x-ttfc-ms" in res2.headers
    finally:
        app.dependency_overrides = {}


def test_structured_json_logging():
    """Verify StructuredJsonFormatter outputs valid JSON containing request_id and metadata."""
    formatter = StructuredJsonFormatter()
    logger = logging.getLogger("test_logger")
    token = request_id_ctx_var.set("req-test-trace-12345")
    try:
        record = logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "Structured test message", (), None)
        out = formatter.format(record)
        data = json.loads(out)
        assert data["level"] == "INFO"
        assert data["message"] == "Structured test message"
        assert data["request_id"] == "req-test-trace-12345"
        assert "timestamp" in data
    finally:
        request_id_ctx_var.reset(token)

    # Test setup_logging executes cleanly
    setup_logging(json_logs=True)
    setup_logging(json_logs=False)


@pytest.mark.asyncio
async def test_telemetry_metrics_tracking_and_endpoint():
    """Verify telemetry collects metrics and exposes them at /metrics and /v1/telemetry."""
    p = MockProvider("openai")
    app.dependency_overrides[get_providers] = lambda: [p]
    telemetry = get_telemetry()
    telemetry.reset()

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
            )
            assert res.status_code == 200

            # Query /metrics
            metrics_res = await ac.get("/metrics")
            assert metrics_res.status_code == 200
            data = metrics_res.json()
            assert data["total_requests"] >= 1
            assert data["successful_requests"] >= 1
            assert data["ttfc_ms"]["count"] >= 1
            assert data["requests_by_provider"].get("openai") >= 1

            # Query /v1/telemetry
            telem_res = await ac.get("/v1/telemetry")
            assert telem_res.status_code == 200
            assert telem_res.json()["total_requests"] == data["total_requests"]
    finally:
        app.dependency_overrides = {}


# ----------------------------------------------------------------------
# 3. Proposal 9: Circuit Breaker Outage Protection
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions():
    """Verify circuit breaker trips to OPEN on failures and recovers."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.2)
    provider = "flaky_provider"

    assert cb.can_execute(provider) is True
    assert cb.get_state(provider) == CircuitState.CLOSED

    # Record 2 failures - still closed
    cb.record_failure(provider)
    cb.record_failure(provider)
    assert cb.can_execute(provider) is True
    assert cb.get_state(provider) == CircuitState.CLOSED

    # 3rd failure trips to OPEN
    cb.record_failure(provider)
    assert cb.get_state(provider) == CircuitState.OPEN
    assert cb.can_execute(provider) is False

    # Wait for recovery timeout
    await asyncio.sleep(0.25)
    # Should transition to HALF_OPEN and allow one canary probe
    assert cb.can_execute(provider) is True
    assert cb.get_state(provider) == CircuitState.HALF_OPEN

    # Success restores to CLOSED
    cb.record_success(provider)
    assert cb.get_state(provider) == CircuitState.CLOSED
    assert cb.can_execute(provider) is True


@pytest.mark.asyncio
async def test_router_manager_circuit_breaker_fail_fast():
    """Verify RouterManager reorders candidates when circuit is OPEN to avoid failing provider."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
    p_bad = MockProvider("bad_openai", fail=True)
    p_good = MockProvider("good_anthropic", fail=False)

    manager = RouterManager([p_bad, p_good], circuit_breaker=cb)

    # Trip the bad provider's circuit
    cb.record_failure("bad_openai")
    cb.record_failure("bad_openai")
    assert cb.can_execute("bad_openai") is False

    # Execute request: good_anthropic should be promoted to primary candidate
    chunks = []
    async for chunk in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello from good_anthropic"
    assert manager.last_selected_provider == "good_anthropic"


@pytest.mark.asyncio
async def test_circuit_breaker_bypasses_open_provider_immediately():
    """Verify that when a provider's circuit is OPEN, RouterManager bypasses it in <0.05s instead of waiting."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    # Provider that delays 5 seconds if called
    p_slow_bad = MockProvider("slow_bad", fail=True, delay=5.0)
    p_fast_good = MockProvider("fast_good", fail=False, delay=0.0)

    # Trip the slow provider's circuit
    cb.record_failure("slow_bad")
    assert cb.can_execute("slow_bad") is False

    manager = RouterManager([p_slow_bad, p_fast_good], circuit_breaker=cb)

    start = asyncio.get_running_loop().time()
    chunks = []
    async for chunk in manager.stream_with_fallback([{"role": "user", "content": "hello"}], preference="slow_bad"):
        chunks.append(chunk)
    elapsed = asyncio.get_running_loop().time() - start

    # Should execute fast_good immediately without incurring slow_bad's 5s delay
    assert elapsed < 0.2
    assert "".join(chunks) == "Hello from fast_good"
    assert manager.last_selected_provider == "fast_good"


@pytest.mark.asyncio
async def test_circuit_breaker_all_providers_open_returns_503():
    """Verify that when all candidate providers have OPEN circuits, endpoint returns HTTP 503."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    p1 = MockProvider("prov1", fail=True)
    p2 = MockProvider("prov2", fail=True)

    cb.record_failure("prov1")
    cb.record_failure("prov2")
    assert cb.can_execute("prov1") is False
    assert cb.can_execute("prov2") is False

    mock_manager = RouterManager([p1, p2], circuit_breaker=cb)

    from app.api.v1.chat import get_router_manager
    app.dependency_overrides[get_router_manager] = lambda: mock_manager

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
            )
            assert res.status_code == 503
            assert "Service Unavailable" in res.json().get("detail", "")
    finally:
        app.dependency_overrides = {}


# ----------------------------------------------------------------------
# 4. Proposal 10: Pluggable Metrics Store
# ----------------------------------------------------------------------

def test_in_memory_metrics_store_synchronization():
    """Verify multiple strategies sharing a store access synchronized metrics."""
    shared_store = InMemoryMetricsStore()

    s1 = LeastInFlightStrategy(store=shared_store)
    s2 = LeastInFlightStrategy(store=shared_store)

    p1 = MockProvider("p1")
    p2 = MockProvider("p2")

    # Increment via s1
    s1.on_request_start("p1")
    # s2 should immediately see the updated in-flight count
    assert s2.store.get_in_flight("p1") == 1
    # Routing via s2 should prefer p2
    assert s2.select_provider([p1, p2]).get_provider_name() == "p2"

    s1.on_request_end("p1")
    assert s2.store.get_in_flight("p1") == 0


def test_redis_metrics_store_with_mock():
    """Verify RedisMetricsStore delegates properly to redis client when available."""
    store = RedisMetricsStore(redis_url="redis://localhost:6379/0")

    mock_client = MagicMock()
    mock_storage = {}

    def mock_get(key):
        return mock_storage.get(key)

    def mock_set(key, val):
        mock_storage[key] = str(val)

    def mock_incr(key):
        val = int(mock_storage.get(key, 0)) + 1
        mock_storage[key] = str(val)
        return val

    def mock_decr(key):
        val = max(0, int(mock_storage.get(key, 0)) - 1)
        mock_storage[key] = str(val)
        return val

    mock_client.get.side_effect = mock_get
    mock_client.set.side_effect = mock_set
    mock_client.incr.side_effect = mock_incr
    mock_client.decr.side_effect = mock_decr
    mock_client.keys.return_value = []

    store._client = mock_client

    assert store.get_in_flight("openai") == 0
    assert store.increment_in_flight("openai") == 1
    assert store.get_in_flight("openai") == 1
    assert store.decrement_in_flight("openai") == 0

    store.set_cost("openai", 0.005)
    assert store.get_cost("openai") == 0.005

    store.update_latency("openai", 0.4)
    assert store.get_latency("openai") == 0.4

    store.update_error_rate("openai", True)
    assert store.get_error_rate("openai") == 1.0

    store.reset()


# ----------------------------------------------------------------------
# 5. Proposal 6: Resource Cleanup
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_close():
    """Verify provider close() cleans up resources without raising."""
    p_mock = MockProvider("test")
    await p_mock.close()
    assert p_mock.closed is True


@pytest.mark.asyncio
async def test_fastapi_lifespan_closes_providers():
    """Verify that FastAPI lifespan teardown calls close() on all providers."""
    p1 = MockProvider("test1")
    p2 = MockProvider("test2")

    app.dependency_overrides[get_providers] = lambda: [p1, p2]
    try:
        async with lifespan(app):
            assert p1.closed is False
            assert p2.closed is False
        # Exiting lifespan triggers teardown
        assert p1.closed is True
        assert p2.closed is True
    finally:
        app.dependency_overrides = {}


