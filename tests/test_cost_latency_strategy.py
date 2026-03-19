import pytest
from app.services.routing.strategies import CostLatencyTradeoffStrategy
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict

class MockProvider(LLMProvider):
    def __init__(self, name: str):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        yield "hello"
    
    def get_provider_name(self) -> str:
        return self.name

def test_cost_latency_strategy_latency():
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    # Only alpha (latency) matters
    strategy = CostLatencyTradeoffStrategy(alpha=1.0, beta=0.0, gamma=0.0)
    
    strategy.update_metrics("p1", latency=0.1)
    strategy.update_metrics("p2", latency=0.5)
    assert strategy.select_provider([p1, p2]) == p1
    
    # Update p1 multiple times to increase its rolling average latency
    for _ in range(10):
        strategy.update_metrics("p1", latency=1.0)
    assert strategy.select_provider([p1, p2]) == p2

def test_cost_latency_strategy_cost():
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    # Only beta (cost) matters
    strategy = CostLatencyTradeoffStrategy(alpha=0.0, beta=1.0, gamma=0.0)
    
    strategy.update_metrics("p1", cost=0.01)
    strategy.update_metrics("p2", cost=0.001)
    assert strategy.select_provider([p1, p2]) == p2
    
    strategy.update_metrics("p1", cost=0.0001)
    assert strategy.select_provider([p1, p2]) == p1

def test_cost_latency_strategy_error_rate():
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    # Only gamma (error rate) matters
    strategy = CostLatencyTradeoffStrategy(alpha=0.0, beta=0.0, gamma=1.0)
    
    strategy.update_metrics("p1", is_error=False)
    strategy.update_metrics("p2", is_error=True)
    assert strategy.select_provider([p1, p2]) == p1
    
    # After multiple errors, p1 should lose
    for _ in range(10):
        strategy.update_metrics("p1", is_error=True)
    strategy.update_metrics("p2", is_error=False)
    assert strategy.select_provider([p1, p2]) == p2

def test_cost_latency_strategy_composite():
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    # Balanced weights
    strategy = CostLatencyTradeoffStrategy(alpha=0.4, beta=0.4, gamma=0.2)
    
    # p1: fast but expensive
    # p2: slow but cheap
    strategy.update_metrics("p1", latency=0.1, cost=0.1, is_error=False)
    strategy.update_metrics("p2", latency=1.0, cost=0.001, is_error=False)
    
    # p1 score: 0.4*(1/0.1) + 0.4*(1/0.1) + 0.2*(1) = 4 + 4 + 0.2 = 8.2
    # p2 score: 0.4*(1/1.0) + 0.4*(1/0.001) + 0.2*(1) = 0.4 + 400 + 0.2 = 400.6
    assert strategy.select_provider([p1, p2]) == p2
    
    # Now make p2 extremely slow
    strategy.update_metrics("p2", latency=1000.0)
    # p2 score: 0.4*(1/1000) + 400 + 0.2 = 0.0004 + 400 + 0.2 = 400.2004
    # Still p2 because cost is so much lower.
    
    # Make p1 extremely cheap
    strategy.update_metrics("p1", cost=0.0001)
    # p1 score: 0.4*(10) + 0.4*(10000) + 0.2 = 4 + 4000 + 0.2 = 4004.2
    assert strategy.select_provider([p1, p2]) == p1
