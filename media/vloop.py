"""A virtual-time asyncio event loop (discrete-event execution of ordinary asyncio code).

`loop.time()` returns a virtual clock. Instead of blocking, the selector jumps the clock to the next
scheduled timer, so `asyncio.sleep`, `call_at` etc. take zero wall time and behave deterministically. There
must be no real I/O and no threads delivering results via `call_soon_threadsafe`. If nothing is scheduled and
nothing is ready, that is a deadlock, and it raises.

Used by `media.session` (mode="virtual") to make the media campaign independent of host load. Real computation
(codec, STT, vision) still runs, and its measured CPU time is charged to virtual time by `Runtime.compute`.
"""

from __future__ import annotations

import asyncio
import selectors


class _VirtualSelector(selectors.BaseSelector):
    def __init__(self) -> None:
        self._map: dict = {}
        self.loop: VirtualTimeLoop | None = None

    def register(self, fileobj, events, data=None):
        key = selectors.SelectorKey(
            fileobj, fileobj if isinstance(fileobj, int) else fileobj.fileno(), events, data
        )
        self._map[fileobj] = key
        return key

    def unregister(self, fileobj):
        return self._map.pop(fileobj)

    def select(self, timeout=None):
        if timeout is None:
            raise RuntimeError("virtual-time loop deadlock: nothing scheduled")
        self.loop._vt += max(0.0, timeout)
        return []

    def get_map(self):
        return self._map

    def close(self):
        self._map.clear()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, start: float = 1000.0) -> None:
        sel = _VirtualSelector()
        self._vt = start
        super().__init__(selector=sel)
        sel.loop = self

    def time(self) -> float:
        return self._vt


def run_virtual(coro, start: float = 1000.0):
    loop = VirtualTimeLoop(start)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
