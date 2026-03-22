"""
core/push.py
Shared websocket push state — extracted from server.py so other modules
can import _connected and _push_to_all without re-executing server.py's
module-level code (which includes the ChromaDB health check).

server.py imports from here instead of defining these itself.
switch_host.py and agent.py import from here instead of from server.
"""

import json

# All currently connected websockets
_connected: set = set()


async def _push_to_all(message: str) -> None:
    """Push a message to every connected client."""
    if not _connected:
        print(f"[Push] Skipped — no clients connected: {message[:60]}")
        return
    payload = json.dumps({"type": "token", "data": message})
    done = json.dumps({"type": "done", "data": ""})
    for ws in list(_connected):
        try:
            await ws.send(payload)
            await ws.send(done)
        except Exception as e:
            print(f"[Push] Failed for client: {e}")
