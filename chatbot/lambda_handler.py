"""
lambda_handler.py
AWS Lambda entry point for the Gizmo conversational API.

Receives POST requests from API Gateway, runs the agent,
and returns the response. Streaming is handled via
API Gateway's chunked transfer — each token is buffered
and returned as a complete response since Lambda's response
model doesn't support true token streaming through API Gateway
without WebSockets. For a personal project this is fine.

Request body (JSON):
    {
        "message":      "user message text",
        "session_id":   "optional session identifier",
        "context": {
            "current_host": "name",
            "fronters":     ["name1", "name2"]
        }
    }

Response body (JSON):
    {
        "response":    "Gizmo's response text",
        "session_id":  "session identifier"
    }

Environment variables (set in Lambda console):
    CHROMA_PERSIST_DIR   — path to ChromaDB on /tmp or mounted EFS
    GOOGLE_DRIVE_SYNC    — "true" to enable Drive sync
    AWS_REGION           — AWS region for Bedrock
    BEDROCK_MODEL_ID     — Bedrock model ID
    BRAVE_API_KEY        — for web search
"""

import json
import os
import uuid

# Ensure dependencies are importable from the Lambda package
import sys
sys.path.insert(0, "/var/task")


def handler(event, context):
    """
    Lambda handler. Synchronous wrapper around the async agent.
    """
    import asyncio

    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return _response(400, {"error": "Invalid JSON body"})

    message = body.get("message", "").strip()
    if not message:
        return _response(400, {"error": "No message provided"})

    session_id = body.get("session_id") or str(uuid.uuid4())
    ctx = body.get("context", {})

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _run_agent(message, session_id, ctx)
        )
        loop.close()
    except Exception as e:
        print(f"[Lambda] Agent error: {e}")
        return _response(500, {"error": str(e)})

    return _response(200, {
        "response":   result,
        "session_id": session_id,
    })


async def _run_agent(message: str, session_id: str, ctx: dict) -> str:
    """
    Run the Gizmo agent and collect the full response.
    """
    # Sync Google Drive before processing if enabled
    if os.getenv("GOOGLE_DRIVE_SYNC") == "true":
        await _sync_from_drive()

    from core.agent import agent
    from memory.history import get_session

    history = get_session(session_id)

    response_chunks = []
    async for chunk in agent.run(
        user_message=message,
        history=history,
        session_id=session_id,
        context=ctx,
    ):
        response_chunks.append(chunk)

    response = "".join(response_chunks)

    # Sync Google Drive after processing if enabled
    if os.getenv("GOOGLE_DRIVE_SYNC") == "true":
        await _sync_to_drive()

    return response


async def _sync_from_drive():
    """Pull latest ChromaDB files from Google Drive before processing."""
    try:
        from persistence.gdrive import pull_chroma
        await pull_chroma()
        print("[Lambda] Pulled ChromaDB from Drive")
    except Exception as e:
        print(f"[Lambda] Drive pull failed (non-fatal): {e}")


async def _sync_to_drive():
    """Push ChromaDB files to Google Drive after processing."""
    try:
        from persistence.gdrive import push_chroma
        await push_chroma()
        print("[Lambda] Pushed ChromaDB to Drive")
    except Exception as e:
        print(f"[Lambda] Drive push failed (non-fatal): {e}")


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
