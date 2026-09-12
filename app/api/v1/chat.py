from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm.base import LLMProvider
from app.services.llm.openai import OpenAIProvider
from app.services.llm.anthropic import AnthropicProvider
from app.services.routing.manager import RouterManager
from app.services.routing.circuit_breaker import CircuitBreakerOpenError
from app.services.telemetry import get_telemetry
from app.core.config import settings
from app.core.middleware import get_request_id
from functools import lru_cache
from typing import List, Optional
import asyncio

router = APIRouter(prefix="/v1")

# In-memory provider discovery
@lru_cache(maxsize=1)
def get_providers() -> List[LLMProvider]:
    providers = []
    has_openai = settings.OPENAI_API_KEY and (
        settings.OPENAI_API_KEY.get_secret_value()
        if hasattr(settings.OPENAI_API_KEY, "get_secret_value")
        else str(settings.OPENAI_API_KEY)
    )
    if has_openai:
        providers.append(OpenAIProvider())

    has_anthropic = settings.ANTHROPIC_API_KEY and (
        settings.ANTHROPIC_API_KEY.get_secret_value()
        if hasattr(settings.ANTHROPIC_API_KEY, "get_secret_value")
        else str(settings.ANTHROPIC_API_KEY)
    )
    if has_anthropic:
        providers.append(AnthropicProvider())
    
    # §2.5: Fail fast when no provider API keys are configured.
    # Silently adding unauthenticated providers wastes TTFC_TIMEOUT × N seconds
    # before every request fails with an auth error. Raising here surfaces the
    # misconfiguration immediately at request time.
    if not providers:
        raise ValueError(
            "No LLM provider API keys are configured. "
            "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in your environment."
        )

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
    req_id = get_request_id()
    telemetry = get_telemetry()
    telemetry.record_request_start()
    start_time = asyncio.get_running_loop().time()
    
    stream_iter = manager.stream_with_fallback(
        messages_dict, 
        request.model_preference, 
        request.fallback_models,
        request.routing_strategy,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
    ).__aiter__()

    # Phase 1: Pre-fetch first chunk before committing HTTP 200 headers to client
    try:
        first_chunk = await stream_iter.__anext__()
        ttfc_ms = int((asyncio.get_running_loop().time() - start_time) * 1000)
    except StopAsyncIteration:
        first_chunk = None
        ttfc_ms = int((asyncio.get_running_loop().time() - start_time) * 1000)
    except Exception as exc:
        total_err_ms = int((asyncio.get_running_loop().time() - start_time) * 1000)
        telemetry.record_request_complete(total_err_ms, success=False)
        if isinstance(exc, CircuitBreakerOpenError) or "circuit breaker" in str(exc).lower():
            raise HTTPException(status_code=503, detail=f"Service Unavailable: {exc}")
        # §2.5: No providers configured → 503 immediately (no timeout wasted).
        if isinstance(exc, ValueError) and "api key" in str(exc).lower():
            raise HTTPException(status_code=503, detail=f"Service Unavailable: {exc}")
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in str(exc).lower():
            raise HTTPException(status_code=504, detail=f"Gateway Timeout: {exc}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {exc}")

    # If stream=False: collect all chunks and return JSON ChatResponse
    if not request.stream:
        chunks = [first_chunk] if first_chunk is not None else []
        async for chunk in stream_iter:
            chunks.append(chunk)
        total_ms = int((asyncio.get_running_loop().time() - start_time) * 1000)
        
        response_model = ChatResponse(
            content="".join(chunks),
            provider=manager.last_selected_provider or "unknown",
            model=manager.last_selected_model
        )
        json_headers = {
            "X-TTFC-Ms": str(ttfc_ms),
            "X-Total-Duration-Ms": str(total_ms),
            "X-LLM-Provider": manager.last_selected_provider or "unknown",
        }
        if req_id:
            json_headers["X-Request-ID"] = req_id
        if manager.last_selected_model:
            json_headers["X-LLM-Model"] = manager.last_selected_model
        return JSONResponse(content=response_model.model_dump(), headers=json_headers)

    # If stream=True: stream first chunk then remaining chunks
    async def stream_generator():
        if first_chunk is not None:
            yield first_chunk
        async for chunk in stream_iter:
            yield chunk

    headers = {
        "X-TTFC-Ms": str(ttfc_ms),
    }
    if req_id:
        headers["X-Request-ID"] = req_id
    if manager.last_selected_provider:
        headers["X-LLM-Provider"] = manager.last_selected_provider
    if manager.last_selected_model:
        headers["X-LLM-Model"] = manager.last_selected_model

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers=headers
    )


