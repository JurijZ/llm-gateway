import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.v1.chat import get_providers
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict

class MockProvider(LLMProvider):
    def __init__(self, name: str):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        yield f"Hello from {self.name}"
    
    def get_provider_name(self) -> str:
        return self.name

# Mock get_providers to return our mock providers
def mock_get_providers():
    return [MockProvider("openai"), MockProvider("anthropic")]

app.dependency_overrides[get_providers] = mock_get_providers

@pytest.mark.asyncio
async def test_chat_with_routing_strategy():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Test with hardcoded strategy and preference
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model_preference": "anthropic",
            "routing_strategy": "hardcoded"
        }
        response = await ac.post("/v1/chat", json=payload)
        assert response.status_code == 200
        assert "Hello from anthropic" in response.text

        # 2. Test with default strategy (hardcoded) and no preference
        payload = {
            "messages": [{"role": "user", "content": "hi"}]
        }
        response = await ac.post("/v1/chat", json=payload)
        assert response.status_code == 200
        assert "Hello from openai" in response.text

# Clear overrides after test
def teardown_module(module):
    app.dependency_overrides = {}
