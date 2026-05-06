"""
core/llm.py
HuggingFace inference via the Inference Router.
Uses the OpenAI-compatible endpoint at router.huggingface.co/v1.

Retry logic:
  - generate() retries up to MAX_RETRIES times on empty response
  - Each retry waits RETRY_DELAY seconds
  - Timeout per attempt: TIMEOUT_SECONDS
  - On exhausted retries, returns empty string — caller handles gracefully
"""

import asyncio
import os
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN      = os.getenv("HF_TOKEN")
HF_MODEL_ID   = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-V3-0324")
HF_ROUTER_URL = "https://router.huggingface.co/v1"

MAX_RETRIES     = 3
RETRY_DELAY     = 1.5   # seconds between retries
TIMEOUT_SECONDS = 45    # per-attempt timeout


class LLMClient:
    def __init__(self, model_id: str = HF_MODEL_ID):
        self.model_id = model_id
        self.client = AsyncOpenAI(
            base_url=HF_ROUTER_URL,
            api_key=HF_TOKEN,
            timeout=TIMEOUT_SECONDS,
        )

    async def stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens. Single attempt — caller handles retries if needed.
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

    async def _attempt_generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Single generate attempt with timeout."""
        result = ""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                async for token in self.stream(
                    messages, system_prompt, max_new_tokens, temperature
                ):
                    result += token
        except asyncio.TimeoutError:
            print(f"[LLM] Attempt timed out after {TIMEOUT_SECONDS}s")
        except Exception as e:
            print(f"[LLM] Attempt failed: {e}")
        return result

    async def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Non-streaming generation with retry on empty response.
        Retries up to MAX_RETRIES times before giving up.
        Returns empty string if all attempts fail — caller handles gracefully.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            result = await self._attempt_generate(
                messages, system_prompt, max_new_tokens, temperature
            )

            if result.strip():
                if attempt > 1:
                    print(f"[LLM] Succeeded on attempt {attempt}")
                return result

            if attempt < MAX_RETRIES:
                print(f"[LLM] Empty response on attempt {attempt}/{MAX_RETRIES} — retrying in {RETRY_DELAY}s")
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"[LLM] All {MAX_RETRIES} attempts returned empty — giving up")

        return ""


# Singleton
llm = LLMClient()