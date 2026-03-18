from typing import AsyncGenerator, List, Dict
import openai
from app.services.llm.base import LLMProvider
from app.core.config import settings


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, default_model: str = "gpt-4o"):

        # print("Default model set to:", default_model)

        self.client = openai.AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.default_model = default_model

    async def stream_chat(
        self, messages: List[Dict[str, str]], model: str = None
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.responses.create(
            model=model or self.default_model,
            input=messages,
            stream=True,
        )
        async for event in stream:
            if event.type == "response.output_text.delta":
                #print(event.delta)
                yield event.delta

    def get_provider_name(self) -> str:
        return "openai"