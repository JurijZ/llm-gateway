from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest
from app.services.llm.openai import OpenAIProvider
from app.services.llm.anthropic import AnthropicProvider
from app.services.routing.manager import RouterManager
from app.core.config import settings
from typing import List

router = APIRouter(prefix="/v1")

# In-memory router manager setup
# In a real app, this could be a global dependency or a singleton
def get_router_manager():
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
        
    return RouterManager(providers)

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, manager: RouterManager = Depends(get_router_manager)):
    # Standardize messages to list of dicts for providers
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
    async def stream_generator():
        async for chunk in manager.stream_with_fallback(messages_dict, request.model_preference):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")
