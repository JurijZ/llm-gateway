import pytest
from app.services.routing.strategies import HardcodedStrategy
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict

class MockProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        yield f"Hello from {self.name}"
    
    def get_provider_name(self) -> str:
        return self.name

def test_hardcoded_strategy():
    p1 = MockProvider("openai")
    p2 = MockProvider("anthropic")
    strategy = HardcodedStrategy()
    
    # Test default
    selected = strategy.select_provider([p1, p2])
    assert selected.get_provider_name() == "openai"
    
    # Test preference
    selected = strategy.select_provider([p1, p2], preference="anthropic")
    assert selected.get_provider_name() == "anthropic"
    
    # Test non-existent preference
    selected = strategy.select_provider([p1, p2], preference="google")
    assert selected.get_provider_name() == "openai"
