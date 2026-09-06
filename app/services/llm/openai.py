from typing import AsyncGenerator, List, Dict, Optional
import openai
from app.services.llm.base import LLMProvider
from app.core.config import settings


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o"):
        if api_key:
            key = api_key
        elif settings.OPENAI_API_KEY:
            key = settings.OPENAI_API_KEY.get_secret_value() if hasattr(settings.OPENAI_API_KEY, "get_secret_value") else str(settings.OPENAI_API_KEY)
        else:
            key = None

        self.client = openai.AsyncOpenAI(api_key=key)
        self.default_model = default_model

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        req_kwargs = {
            "model": model or self.default_model,
            "input": messages,
            "stream": True,
        }
        if temperature is not None:
            req_kwargs["temperature"] = temperature
        if max_tokens is not None:
            req_kwargs["max_output_tokens"] = max_tokens

        response_stream = await self.client.responses.create(**req_kwargs)
        async with response_stream as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    async def close(self) -> None:
        await self.client.close()

    def get_provider_name(self) -> str:
        return "openai"
