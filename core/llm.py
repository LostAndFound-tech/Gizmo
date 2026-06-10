"""
core/llm.py
Main LLM client — OpenRouter with DeepSeek V4 Flash.

Migrated from HuggingFace inference router to OpenRouter.
Model is configurable via OPENROUTER_MODEL_ID env var.
Default: deepseek/deepseek-v4-flash

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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL_ID", "deepseek/deepseek-v4-flash")

MAX_RETRIES     = 5
RETRY_DELAY     = .5   # slightly longer — OpenRouter occasionally needs it
TIMEOUT_SECONDS = 60    # increased from 45 — V4 Flash can be slower on long prompts


class LLMClient:
    def __init__(self, model_id: str = OPENROUTER_MODEL):
        self.model_id = model_id
        self.client = AsyncOpenAI(
            base_url=OPENROUTER_URL,
            api_key=OPENROUTER_API_KEY,
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
