import pytest
from app.services.routing.manager import RouterManager
from app.services.llm.base import LLMProvider
from typing import AsyncGenerator, List, Dict
import asyncio

class FailingProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        raise Exception(f"{self.name} failed")
        yield "never"
    
    def get_provider_name(self) -> str:
        return self.name

class WorkingProvider(LLMProvider):
    def __init__(self, name):
        self.name = name
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str = None) -> AsyncGenerator[str, None]:
        yield f"Hello from {self.name}"
    
    def get_provider_name(self) -> str:
        return self.name

@pytest.mark.asyncio
async def test_router_manager_fallback():
    p1 = FailingProvider("failing-openai")
    p2 = WorkingProvider("working-anthropic")
    
    manager = RouterManager([p1, p2])
    
    chunks = []
    async for chunk in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)
    
    assert "".join(chunks) == "Hello from working-anthropic"

@pytest.mark.asyncio
async def test_router_manager_all_fail():
    p1 = FailingProvider("fail1")
    p2 = FailingProvider("fail2")
    
    manager = RouterManager([p1, p2])
    
    with pytest.raises(Exception) as excinfo:
        async for _ in manager.stream_with_fallback([{"role": "user", "content": "hi"}]):
            pass
    assert "fail2 failed" in str(excinfo.value)
