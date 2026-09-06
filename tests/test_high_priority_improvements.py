import pytest
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator, List, Dict, Optional
import asyncio

from app.main import app
from app.api.v1.chat import get_providers
from app.services.llm.base import LLMProvider
from app.services.routing.strategies import RoutingStrategy, CostLatencyTradeoffStrategy
from app.services.routing.manager import RouterManager
from app.core.models import get_model_cost, MODEL_PRICING

class DummyProvider(LLMProvider):
    def __init__(self, name: str, chunks: Optional[List[str]] = None, fail_before_first: bool = False, timeout_before_first: bool = False):
        self.name = name
        self.chunks = chunks or [f"chunk1 from {name}", f" chunk2 from {name}"]
        self.fail_before_first = fail_before_first
        self.timeout_before_first = timeout_before_first

    async def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> AsyncGenerator[str, None]:
        if self.timeout_before_first:
            raise asyncio.TimeoutError(f"Connection to {self.name} timed out")
        if self.fail_before_first:
            raise RuntimeError(f"Provider {self.name} connection refused")
        for c in self.chunks:
            yield c

    def get_provider_name(self) -> str:
        return self.name

@pytest.mark.asyncio
async def test_non_streaming_response():
    """Test Proposal 2: non-streaming chat endpoint returns JSON ChatResponse."""
    p1 = DummyProvider("openai", ["Hello, ", "world!"])
    app.dependency_overrides[get_providers] = lambda: [p1]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            payload = {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "model_preference": "gpt-4o"
            }
            response = await ac.post("/v1/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Hello, world!"
            assert data["provider"] == "openai"
            assert data["model"] == "gpt-4o"
    finally:
        app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_precommit_error_handling_502():
    """Test Proposal 5: all providers failing before first chunk returns HTTP 502."""
    p1 = DummyProvider("openai", fail_before_first=True)
    p2 = DummyProvider("anthropic", fail_before_first=True)
    app.dependency_overrides[get_providers] = lambda: [p1, p2]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            payload = {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True
            }
            response = await ac.post("/v1/chat", json=payload)
            assert response.status_code == 502
            assert "Bad Gateway" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_precommit_timeout_handling_504():
    """Test Proposal 5: upstream timeout before first chunk returns HTTP 504."""
    p1 = DummyProvider("openai", timeout_before_first=True)
    app.dependency_overrides[get_providers] = lambda: [p1]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            payload = {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True
            }
            response = await ac.post("/v1/chat", json=payload)
            assert response.status_code == 504
            assert "Gateway Timeout" in response.json()["detail"]
    finally:
        app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_strategy_polymorphic_lifecycle_hooks():
    """Test Proposal 3: lifecycle hooks are called without isinstance checks."""
    events = []

    class SpyStrategy(RoutingStrategy):
        def select_provider(self, providers: List[LLMProvider], preference: Optional[str] = None) -> LLMProvider:
            return providers[0]

        def on_request_start(self, provider_name: str, model: Optional[str] = None) -> None:
            events.append(("start", provider_name, model))

        def on_first_chunk(self, provider_name: str, latency: float, model: Optional[str] = None) -> None:
            events.append(("first_chunk", provider_name, latency > 0))

        def on_request_success(self, provider_name: str, model: Optional[str] = None, **kwargs) -> None:
            events.append(("success", provider_name))

        def on_request_end(self, provider_name: str, model: Optional[str] = None) -> None:
            events.append(("end", provider_name))

    spy = SpyStrategy()
    p1 = DummyProvider("openai", ["chunk1", "chunk2"])
    manager = RouterManager([p1])

    # Execute stream with spy strategy directly
    # Monkey-patch get_strategy to return spy
    from app.services.routing import manager as mgr_module
    orig_get_strategy = mgr_module.get_strategy
    mgr_module.get_strategy = lambda _: spy

    try:
        chunks = []
        async for chunk in manager.stream_with_fallback([{"role": "user", "content": "hi"}], strategy_type="spy"):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        # Verify event sequence
        assert ("start", "openai", None) in events
        assert any(e[0] == "first_chunk" and e[1] == "openai" for e in events)
        assert ("success", "openai") in events
        assert ("end", "openai") in events
    finally:
        mgr_module.get_strategy = orig_get_strategy

def test_model_cost_pricing_and_strategy_integration():
    """Test Proposal 4: model pricing integration in CostLatencyTradeoffStrategy."""
    # Verify pricing lookup
    gpt4o_cost = get_model_cost("openai", "gpt-4o")
    mini_cost = get_model_cost("openai", "gpt-4o-mini")
    assert mini_cost < gpt4o_cost

    # Verify CostLatencyTradeoffStrategy uses model pricing
    p_expensive = DummyProvider("openai")
    p_cheap = DummyProvider("anthropic")

    strategy = CostLatencyTradeoffStrategy(alpha=0.0, beta=1.0, gamma=0.0)
    # Give expensive provider gpt-5.4-pro cost, and cheap provider haiku cost
    strategy.on_first_chunk("openai", latency=0.1, model="gpt-5.4-pro")
    strategy.on_first_chunk("anthropic", latency=0.1, model="claude-4-5-haiku")

    # anthropic (haiku) is significantly cheaper, so it should be selected by cost strategy
    selected = strategy.select_provider([p_expensive, p_cheap])
    assert selected.get_provider_name() == "anthropic"

