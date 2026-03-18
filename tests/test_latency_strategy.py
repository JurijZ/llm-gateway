import pytest
from app.services.routing.strategies import LatencyBasedStrategy
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict

class MockProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        yield f"Hello from {self.name}"
    
    def get_provider_name(self) -> str:
        return self.name

def test_latency_strategy():
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    strategy = LatencyBasedStrategy()
    
    # Initially returns first
    assert strategy.select_provider([p1, p2]).get_provider_name() == "p1"
    
    # Update latencies
    strategy.update_latency("p1", 100) # slower
    strategy.update_latency("p2", 50)  # faster
    
    assert strategy.select_provider([p1, p2]).get_provider_name() == "p2"
    
    # Update p1 to be faster
    strategy.update_latency("p1", 10)
    # p1: 100 * 0.7 + 10 * 0.3 = 70 + 3 = 73
    # p2: 50
    assert strategy.select_provider([p1, p2]).get_provider_name() == "p2"
    
    # Update p1 again to be much faster
    strategy.update_latency("p1", 10)
    # p1: 73 * 0.7 + 10 * 0.3 = 51.1 + 3 = 54.1
    # p2: 50
    assert strategy.select_provider([p1, p2]).get_provider_name() == "p2"
    
    # Update p1 again
    strategy.update_latency("p1", 10)
    # p1: 54.1 * 0.7 + 10 * 0.3 = 37.87 + 3 = 40.87
    # p2: 50
    assert strategy.select_provider([p1, p2]).get_provider_name() == "p1"
