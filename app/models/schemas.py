from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model_preference: Optional[str] = None
    # Ordered list of model names to try if the primary provider fails.
    # Each entry is resolved the same way as model_preference.
    # The routing strategy still selects the primary; this chain takes over on failure.
    fallback_models: Optional[List[str]] = None
    routing_strategy: Optional[Literal["hardcoded", "load_balance", "latency", "cost_latency"]] = None
    stream: bool = True

class ChatResponse(BaseModel):
    content: str
    provider: str
