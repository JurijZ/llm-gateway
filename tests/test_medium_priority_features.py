import pytest
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator, List, Dict, Optional
import asyncio
from pydantic import SecretStr, ValidationError

from app.main import app
from app.core.config import Settings
from app.models.schemas import ChatRequest, Message
from app.api.v1.chat import get_providers
from app.services.llm.base import LLMProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.anthropic import AnthropicProvider
from app.services.routing.circuit_breaker import CircuitBreaker, CircuitState, get_circuit_breaker
from app.services.routing.store import InMemoryMetricsStore
from app.services.routing.strategies import LeastInFlightStrategy, LatencyBasedStrategy
from app.services.routing.manager import RouterManager

class MockProvider(LLMProvider):
    def __init__(self, name: str, fail: bool = False):
        self.name = name
        self.fail = fail
        self.closed = False
        self.last_temperature = None
        self.last_max_tokens = None

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
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


# ----------------------------------------------------------------------
# 5. Proposal 6: Resource Cleanup
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_close():
    """Verify provider close() cleans up resources without raising."""
    p_mock = MockProvider("test")
    await p_mock.close()
    assert p_mock.closed is True

