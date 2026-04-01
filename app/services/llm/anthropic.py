from typing import AsyncGenerator, List, Dict, Optional
import anthropic
from app.services.llm.base import LLMProvider
from app.core.config import settings

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None, default_model: str = "claude-3-5-sonnet-20240620"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key or settings.ANTHROPIC_API_KEY)
        self.default_model = default_model

    async def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> AsyncGenerator[str, None]:
        # Convert messages from OpenAI format to Anthropic if necessary
        # OpenAI: [{"role": "user", "content": "..."}]
        # Anthropic: [{"role": "user", "content": "..."}]
        # But Anthropic doesn't support "system" message in the list.
        # It's usually a separate parameter.
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        chat_messages = [m for m in messages if m["role"] != "system"]

        async with self.client.messages.stream(
            model=model or self.default_model,
            max_tokens=4096,
            messages=chat_messages,
            system=system_msg
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def get_provider_name(self) -> str:
        return "anthropic"
