"""
client.py
Terminal client for texting Gizmo.

Usage:
    # Local:
    python client.py

    # Render:
    python client.py --host wss://your-app.onrender.com

    # With identity:
    python client.py --host wss://your-app.onrender.com --name alice
"""

import asyncio
import json
import sys
import argparse
import uuid

WS_HOST = "ws://localhost:8765"


async def chat(host: str, name: str, session_id: str):
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        sys.exit(1)

    # Detect local timezone
    try:
        import datetime
        tz = datetime.datetime.now().astimezone().tzname()
        # Get IANA timezone name if tzlocal is available, fall back to offset
        try:
            from tzlocal import get_localzone
            tz = str(get_localzone())
        except ImportError:
            # Fall back to UTC offset string e.g. "UTC-06:00"
            offset = datetime.datetime.now().astimezone().strftime("%z")
            tz = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    except Exception:
        tz = "UTC"

    print(f"\nConnecting to Gizmo at {host}...")
    print(f"Session: {session_id[:8]} | Identity: {name or 'unset'}")
    print("Type your message and press Enter. Ctrl+C to exit.\n")

    async with websockets.connect(host, ping_interval=20) as ws:
        print("Connected.\n")

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
                "message": message,
                "session_id": session_id,
                "context": {
                    "current_host": name,
                    "fronters": [name] if name else [],
                    "timezone": tz,
                } if name else {"timezone": tz},
            }

            await ws.send(json.dumps(payload))

            # Stream response
            print("gizmo: ", end="", flush=True)
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")

                if t == "token":
                    print(msg["data"], end="", flush=True)

                elif t == "done":
                    print()  # newline after response
                    mood = msg.get("mood")
                    if mood and mood.get("emotion") != "Neutral":
                        print(f"  [{mood['emotion']}, {mood['intensity']:.0%}]\n")
                    else:
                        print()
                    break

                elif t == "error":
                    print(f"\n[error: {msg['data']}]\n")
                    break


def main():
    parser = argparse.ArgumentParser(description="Text Gizmo")
    parser.add_argument("--host", default=WS_HOST, help="WebSocket URL")
    parser.add_argument("--name", default="", help="Your name / current fronter")
    parser.add_argument("--session", default="", help="Session ID (auto-generated if omitted)")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    asyncio.run(chat(args.host, args.name, session_id))


if __name__ == "__main__":
    main()
