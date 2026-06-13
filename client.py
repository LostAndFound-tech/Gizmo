"""
client.py
Terminal client for Halfhoun / Gizmo.

Usage:
    # Local:
    python client.py

    # Render:
    python client.py --host wss://your-app.onrender.com

    # With identity:
    python client.py --host wss://your-app.onrender.com --name ara
"""

import asyncio
import json
import sys
import argparse
import uuid


WS_HOST = "ws://localhost:10000"


async def chat(host: str, name: str, session_id: str):
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        sys.exit(1)

    # Detect local timezone
    try:
        import datetime
        try:
            from tzlocal import get_localzone
            tz = str(get_localzone())
        except ImportError:
            offset = datetime.datetime.now().astimezone().strftime("%z")
            tz = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    except Exception:
        tz = "UTC"

    print(f"\nConnecting to Halfhoun at {host}...")
    print(f"Session: {session_id[:8]} | Identity: {name or 'unset'}")
    print("Type your message and press Enter. Ctrl+C to exit.\n")

    async with websockets.connect(host, ping_interval=20) as ws:
        print("Connected.\n")

        # Keepalive ping task
        async def _ping():
            while True:
                await asyncio.sleep(12)
                try:
                    await ws.send(json.dumps({"type": "ping"}))
                except Exception:
                    break

        asyncio.ensure_future(_ping())

        while True:
            try:
                message = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(f"{name or 'you'}: ")
                )
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not message.strip():
                continue

            payload = {
                "type":       "message",
                "session_id": session_id,
                "content":    message,
                "context": {
                    "current_host": name,
                    "fronters":     [name] if name else [],
                    "timezone":     tz,
                } if name else {"timezone": tz},
            }

            await ws.send(json.dumps(payload))

            # Stream response
            response_chunks = []
            print("Halfhoun: ", end="", flush=True)

            async for raw in ws:
                msg = json.loads(raw)
                t   = msg.get("type")

                if t == "pong":
                    continue

                if t == "thinking":
                    continue

                if t == "chunk":
                    chunk = msg.get("content", "")
                    print(chunk, end="", flush=True)
                    response_chunks.append(chunk)

                elif t == "done":
                    print("\n")
                    # Send received ack so server clears pending_response
                    try:
                        await ws.send(json.dumps({
                            "type":       "received",
                            "session_id": session_id,
                        }))
                    except Exception:
                        pass
                    break

                elif t == "error":
                    print(f"\n[error: {msg.get('message', '?')}]\n")
                    break


def main():
    parser = argparse.ArgumentParser(description="Text Halfhoun")
    parser.add_argument("--host",    default=WS_HOST, help="WebSocket URL")
    parser.add_argument("--name",    default="",       help="Your name / current fronter")
    parser.add_argument("--session", default="",       help="Session ID (auto-generated if omitted)")
    args = parser.parse_args()

    session_id = args.session or f"sess_{''.join(str(uuid.uuid4()).split('-'))[:10]}"
    asyncio.run(chat(args.host, args.name, session_id))


if __name__ == "__main__":
    main()
