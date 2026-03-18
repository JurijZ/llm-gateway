from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model_preference: Optional[str] = None
    stream: bool = True

class ChatResponse(BaseModel):
    content: str
    provider: str
