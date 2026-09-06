from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm.base import LLMProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.anthropic import AnthropicProvider
from app.services.routing.manager import RouterManager
from app.core.config import settings
from functools import lru_cache
from typing import List, Optional
import asyncio

router = APIRouter(prefix="/v1")

# In-memory provider discovery
@lru_cache(maxsize=1)
def get_providers() -> List[LLMProvider]:
    providers = []
    if settings.OPENAI_API_KEY:
        providers.append(OpenAIProvider())
    if settings.ANTHROPIC_API_KEY:
        providers.append(AnthropicProvider())
    
    # If no keys are set, add placeholders or raise error
    # For now, we assume keys are set or handled by providers
    if not providers:
        # We can add them anyway, they'll just fail later if keys are missing
        providers = [OpenAIProvider(), AnthropicProvider()]
        
    return providers

def get_router_manager(
    providers: List[LLMProvider] = Depends(get_providers)
) -> RouterManager:
    return RouterManager(providers)

@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    manager: RouterManager = Depends(get_router_manager)
):
    # Standardize messages to list of dicts for providers
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
    stream_iter = manager.stream_with_fallback(
        messages_dict, 
        request.model_preference, 
        request.fallback_models,
        request.routing_strategy
    ).__aiter__()

    # Phase 1: Pre-fetch first chunk before committing HTTP 200 headers to client
    try:
        first_chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except Exception as exc:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in str(exc).lower():
            raise HTTPException(status_code=504, detail=f"Gateway Timeout: {exc}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {exc}")

    # If stream=False: collect all chunks and return JSON ChatResponse
    if not request.stream:
        chunks = [first_chunk] if first_chunk is not None else []
        async for chunk in stream_iter:
            chunks.append(chunk)
        return ChatResponse(
            content="".join(chunks),
            provider=manager.last_selected_provider or "unknown",
            model=manager.last_selected_model
        )

    # If stream=True: stream first chunk then remaining chunks
    async def stream_generator():
        if first_chunk is not None:
            yield first_chunk
        async for chunk in stream_iter:
            yield chunk

    headers = {}
    if manager.last_selected_provider:
        headers["X-LLM-Provider"] = manager.last_selected_provider
    if manager.last_selected_model:
        headers["X-LLM-Model"] = manager.last_selected_model

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers=headers if headers else None
    )

