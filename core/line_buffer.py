"""
core/line_buffer.py

Two-line chunk buffer. Fires on_chunk every time 2 lines accumulate.
Call flush() at end of session to handle a lone final line.

Usage:
    buf = LineBuffer(on_chunk=my_handler)
    buf.push("Ara: She's just...")
    buf.push("Ara: sitting there.")   # fires on_chunk(["Ara: She's just...", "Ara: sitting there."])
    await buf.flush()                  # fires if 1 line is left over
"""

from typing import Callable, Awaitable


class LineBuffer:

    def __init__(self, on_chunk: Callable[[list[str]], Awaitable[None]]):
        self._buffer: list[str] = []
        self._on_chunk = on_chunk

    async def push(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        self._buffer.append(line)
        if len(self._buffer) == 2:
            chunk = self._buffer.copy()
            self._buffer.clear()
            await self._on_chunk(chunk)

    async def flush(self) -> None:
        """Process any remaining line at end of stream."""
        if self._buffer:
            chunk = self._buffer.copy()
            self._buffer.clear()
            await self._on_chunk(chunk)
