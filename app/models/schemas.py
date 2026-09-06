from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=100_000)

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    model_preference: Optional[str] = None
    # Ordered list of model names to try if the primary provider fails.
    # Each entry is resolved the same way as model_preference.
    # The routing strategy still selects the primary; this chain takes over on failure.
    fallback_models: Optional[List[str]] = None
    routing_strategy: Optional[Literal["hardcoded", "load_balance", "latency", "cost_latency"]] = None
    stream: bool = True
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128_000)

class ChatResponse(BaseModel):
    content: str
    provider: str
    model: Optional[str] = None

