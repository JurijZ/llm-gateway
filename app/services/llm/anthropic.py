from typing import AsyncGenerator, List, Dict, Optional
import anthropic
from app.services.llm.base import LLMProvider
from app.core.config import settings

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-3-5-sonnet-20241022"):
        if api_key:
            key = api_key
        elif settings.ANTHROPIC_API_KEY:
            key = settings.ANTHROPIC_API_KEY.get_secret_value() if hasattr(settings.ANTHROPIC_API_KEY, "get_secret_value") else str(settings.ANTHROPIC_API_KEY)
        else:
            key = None

        self.client = anthropic.AsyncAnthropic(api_key=key)
        self.default_model = default_model

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        # Convert messages from OpenAI format to Anthropic if necessary
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        chat_messages = [m for m in messages if m["role"] != "system"]

        req_kwargs = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or 4096,
            "messages": chat_messages,
        }
        if temperature is not None:
            req_kwargs["temperature"] = temperature
        if system_msg:
            req_kwargs["system"] = system_msg

        async with self.client.messages.stream(**req_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        await self.client.close()

    def get_provider_name(self) -> str:
        return "anthropic"

