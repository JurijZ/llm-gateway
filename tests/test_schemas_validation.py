import pytest
from pydantic import ValidationError
from app.models.schemas import ChatRequest, Message

def test_chat_request_valid_strategies():
    msg = Message(role="user", content="hello")
    
    # Valid strategies
    for strategy in ["hardcoded", "load_balance", "latency", "cost_latency", None]:
        req = ChatRequest(messages=[msg], routing_strategy=strategy)
        assert req.routing_strategy == strategy

def test_chat_request_invalid_strategy():
    msg = Message(role="user", content="hello")
    
    # Invalid strategy should raise ValidationError
    with pytest.raises(ValidationError):
        ChatRequest(messages=[msg], routing_strategy="invalid_strategy")

def test_chat_request_default():
    msg = Message(role="user", content="hello")
    req = ChatRequest(messages=[msg])
    assert req.routing_strategy is None
