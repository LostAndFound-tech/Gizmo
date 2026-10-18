"""
client.py
Terminal client for texting Gizmo.

Usage:
    python client.py                                   # local server on :10000
    python client.py --name alice                      # with identity
    python client.py --host wss://your-app.onrender.com/ws --name alice
    python client.py --new                             # fresh session instead of resuming

Sessions are stable per name (local-<name>) so history resumes between runs.
"""

import argparse
import asyncio
import datetime
import json
import sys
import uuid

WS_HOST = "ws://localhost:10000/ws"


def local_tz() -> str:
    try:
        from tzlocal import get_localzone
        return str(get_localzone())
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


async def chat(host: str, name: str, session_id: str):
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        sys.exit(1)

    name = name.strip().lower()
    tz = local_tz()

    print(f"\nConnecting to Gizmo at {host}...")
    print(f"Session: {session_id[:16]} | Identity: {name or 'unset'}")
    print("Type your message and press Enter. Ctrl+C to exit.\n")

    async with websockets.connect(host, ping_interval=20) as ws:
        print("Connected.\n")

        while True:
            try:
                message = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: input(f"{name or 'you'}: ")
                )
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not message.strip():
                continue

            context = {"timezone": tz}
            if name:
                context["current_host"] = name
                context["fronters"] = [name]

            await ws.send(json.dumps({
                "type":       "message",
                "content":    message,
                "session_id": session_id,
                "context":    context,
            }))

            print("gizmo: ", end="", flush=True)
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    t = msg.get("type")

                    if t == "chunk":
                        print(msg.get("content", ""), end="", flush=True)

                    elif t == "done":
                        print("\n")
                        break

                    elif t == "error":
                        print(f"\n[error: {msg.get('message')}]\n")
                        break

                    # "thinking" / "pong" / anything else: ignore
                else:
                    print("\n[connection closed by server]")
                    break
            except websockets.ConnectionClosed:
                print("\n[connection lost]")
                break


def main():
    parser = argparse.ArgumentParser(description="Text Gizmo")
    parser.add_argument("--host", default=WS_HOST, help="WebSocket URL (include /ws)")
    parser.add_argument("--name", default="", help="Your name / current fronter")
    parser.add_argument("--session", default="", help="Explicit session ID")
    parser.add_argument("--new", action="store_true", help="Start a fresh session")
    args = parser.parse_args()

    if args.session:
        session_id = args.session
    elif args.new or not args.name:
        session_id = f"local-{uuid.uuid4().hex[:12]}"
    else:
        session_id = f"local-{args.name.strip().lower()}"

    try:
        asyncio.run(chat(args.host, args.name, session_id))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"Couldn't connect to {args.host}: {e}")


if __name__ == "__main__":
    main()
