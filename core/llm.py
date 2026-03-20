"""
core/llm.py
AWS Bedrock inference client.
Uses Mistral Large 2 via the Bedrock Runtime API.

Supports streaming and non-streaming generation.
Credentials are pulled from environment or IAM role automatically
via boto3's standard credential chain.

Requirements:
    pip install boto3

Environment variables:
    AWS_REGION          — defaults to us-east-1
    BEDROCK_MODEL_ID    — defaults to mistral.mistral-large-2407-v1:0
"""

import json
import os
from typing import AsyncGenerator, Optional
import asyncio
from dotenv import load_dotenv

load_dotenv()

AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "mistral.mistral-large-2407-v1:0"
)


def _get_client():
    """Get a boto3 Bedrock Runtime client. Cached after first call."""
    import boto3
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )


def _build_payload(
    messages: list[dict],
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """
    Build the Bedrock request payload for Mistral models.
    Mistral on Bedrock uses the messages API format.
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    return {
        "messages": full_messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }


class LLMClient:
    def __init__(self, model_id: str = BEDROCK_MODEL_ID):
        self.model_id = model_id

    async def stream(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Bedrock.
        Runs the blocking boto3 call in an executor to avoid
        blocking the event loop.
        """
        payload = _build_payload(messages, system_prompt, max_new_tokens, temperature)
        loop = asyncio.get_event_loop()

        def _invoke_stream():
            client = _get_client()
            return client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        try:
            response = await loop.run_in_executor(None, _invoke_stream)
            stream = response.get("body")

            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes", b"{}"))
                    # Mistral streaming format
                    delta = (
                        chunk_data
                        .get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta

        except Exception as e:
            print(f"[LLM] Stream failed: {e}")
            raise

    async def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Non-streaming generation. Returns full response string.
        Uses invoke_model for slightly lower latency on short responses.
        """
        payload = _build_payload(messages, system_prompt, max_new_tokens, temperature)
        loop = asyncio.get_event_loop()

        def _invoke():
            client = _get_client()
            return client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

        try:
            response = await loop.run_in_executor(None, _invoke)
            body = json.loads(response["body"].read())
            return (
                body
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as e:
            print(f"[LLM] Generate failed: {e}")
            raise


# Singleton
llm = LLMClient()