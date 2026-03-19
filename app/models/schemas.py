from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model_preference: Optional[str] = None
    routing_strategy: Optional[Literal["hardcoded", "load_balance", "latency", "cost_latency"]] = None
    stream: bool = True

class ChatResponse(BaseModel):
    content: str
    provider: str
