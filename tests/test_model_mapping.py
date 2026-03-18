import pytest
from app.core.models import get_model_info, MODEL_MAPPING, PROVIDER_DEFAULTS
from app.services.routing.manager import RouterManager
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict

class MockProvider(LLMProvider):
    def __init__(self, name: str):
        self.name = name
        self.last_model_used = None

    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        self.last_model_used = model
        yield "chunk"

    def get_provider_name(self) -> str:
        return self.name

@pytest.mark.asyncio
async def test_get_model_info():
    # Test specific model
    provider, model = get_model_info("gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"

    # Test provider name
    provider, model = get_model_info("anthropic")
    assert provider == "anthropic"
    assert model == PROVIDER_DEFAULTS["anthropic"]

    # Test unknown
    provider, model = get_model_info("unknown")
    assert provider is None
    assert model is None

@pytest.mark.asyncio
async def test_router_manager_uses_mapped_model():
    openai_mock = MockProvider("openai")
    anthropic_mock = MockProvider("anthropic")
    manager = RouterManager([openai_mock, anthropic_mock], strategy_type="hardcoded")

    # Request gpt-4o
    messages = [{"role": "user", "content": "hi"}]
    async for _ in manager.stream_with_fallback(messages, preference="gpt-4o"):
        pass
    
    assert openai_mock.last_model_used == "gpt-4o"
    
    # Request claude-3-opus
    async for _ in manager.stream_with_fallback(messages, preference="claude-3-opus"):
        pass
        
    assert anthropic_mock.last_model_used == "claude-3-opus-20240229"

@pytest.mark.asyncio
async def test_router_manager_fallback_behavior():
    # If a model preference is provided but that provider is NOT available
    # It should fallback to available providers and NOT pass the model ID (which would be invalid)
    anthropic_mock = MockProvider("anthropic")
    manager = RouterManager([anthropic_mock], strategy_type="hardcoded")
    
    messages = [{"role": "user", "content": "hi"}]
    # Preference is openai model, but only anthropic is available
    async for _ in manager.stream_with_fallback(messages, preference="gpt-4o"):
        pass
        
    assert anthropic_mock.get_provider_name() == "anthropic"
    assert anthropic_mock.last_model_used is None # Should use its own default
