"""
core/llm.py
HuggingFace inference via the Inference Router.
Uses the OpenAI-compatible endpoint at router.huggingface.co/v1.
"""

import os
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-V3-0324")
HF_ROUTER_URL = "https://router.huggingface.co/v1"


class LLMClient:
    def __init__(self, model_id: str = HF_MODEL_ID):
        self.model_id = model_id
        try:
            self.client = AsyncOpenAI(
                base_url=HF_ROUTER_URL,
                api_key=HF_TOKEN,
            )
        except:
            self.client = AsyncOpenAI(
                base_url="silly",
                api_key="goose"
            )

    async def stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens. Accepts a proper messages list for multi-turn conversation.
        Prepends system prompt if provided.
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        stream = await self.client.chat.completions.create(
            model=self.model_id,
            messages=full_messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming generation. Returns full response string."""
        result = ""
        async for token in self.stream(messages, system_prompt, max_new_tokens, temperature):
            result += token
        return result


# Singleton
llm = LLMClient()