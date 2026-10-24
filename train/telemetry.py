"""Training telemetry: per-step time, per-collective time, tokens/s, MFU, memory.

Collective timing: we wrap the `torch.distributed` collective entry points (used by FSDP2) and give DDP a
comm hook. For async ops, completion time comes from a callback on the Work's future, so a measured
duration covers issue → completion, *including* time queued behind earlier collectives on the same
process group.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist

_WRAPPED = [
    "all_gather_single",
    "reduce_scatter_single",
    "all_reduce",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "broadcast",
    "all_gather",
    "reduce_scatter",
    "send",
    "recv",
    "barrier",
]


class CollectiveTimer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, float, float, int]] = []  # (op, t_issue, t_done, bytes)
        self._orig: dict[str, Any] = {}

    def install(self) -> None:
        for name in _WRAPPED:
            fn = getattr(dist, name, None)
            if fn is None or name in self._orig:
                continue
            self._orig[name] = fn
            setattr(dist, name, self._wrap(name, fn))

    def uninstall(self) -> None:
        for name, fn in self._orig.items():
            setattr(dist, name, fn)
        self._orig.clear()

    def _record(self, op: str, t0: float, t1: float, nbytes: int) -> None:
        with self.lock:
            self.events.append((op, t0, t1, nbytes))

    def _wrap(self, name: str, fn):
        timer = self

        def wrapped(*args, **kwargs):
            nbytes = 0
            for a in list(args[:2]) + list(kwargs.values()):
                if isinstance(a, torch.Tensor):
                    nbytes = max(nbytes, a.numel() * a.element_size())
            t0 = time.perf_counter()
            work = fn(*args, **kwargs)
            if kwargs.get("async_op") and work is not None:
                try:
                    fut = work.get_future()
                    fut.add_done_callback(lambda _f: timer._record(name, t0, time.perf_counter(), nbytes))
                except Exception:
                    timer._record(name, t0, time.perf_counter(), nbytes)
            else:
                timer._record(name, t0, time.perf_counter(), nbytes)
            return work

        return wrapped

    def ddp_hook(self):
        """DDP comm hook: allreduce-mean with timing."""
        timer = self

        def hook(state, bucket):
            t = bucket.buffer()
            nbytes = t.numel() * t.element_size()
            t0 = time.perf_counter()
            t.div_(dist.get_world_size())
            # use the unwrapped all_reduce so the bucket is not double-counted as "all_reduce"
            work = timer._orig.get("all_reduce", dist.all_reduce)(t, async_op=True)
            fut = work.get_future()

            def done(f):
                timer._record("ddp_allreduce", t0, time.perf_counter(), nbytes)
                return f.value()[0]

            return fut.then(done)

        return hook

    def drain(self) -> dict[str, dict[str, float]]:
        with self.lock:
            ev, self.events = self.events, []
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "time_s": 0.0, "bytes": 0})
        for op, t0, t1, nb in ev:
            a = agg[op]
            a["n"] += 1
            a["time_s"] += t1 - t0
            a["bytes"] += nb
        return dict(agg)


class TelemetrySink:
    """Writes per-step records to JSONL; optionally also sends them as UDP datagrams to a collector."""

    def __init__(self, path: Path | None, udp: str | None = None, rank: int = 0):
        self.f = open(path, "a", buffering=1) if path else None
        self.rank = rank
        self.sock = None
        if udp:
            host, port = udp.rsplit(":", 1)
            self.addr = (host, int(port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.proc = psutil.Process(os.getpid())

    def emit(self, rec: dict[str, Any]) -> None:
        rec = {"rank": self.rank, "wall": time.time(), "rss_mb": self.proc.memory_info().rss / 2**20, **rec}
        line = json.dumps(rec)
        if self.f:
            self.f.write(line + "\n")
        if self.sock:
            try:
                self.sock.sendto(line.encode()[:60000], self.addr)
            except OSError:
                pass

    def close(self) -> None:
        if self.f:
            self.f.close()


def measure_peak_matmul_flops(threads: int, n: int = 1024, reps: int = 10) -> float:
    """Empirical fp32 matmul peak for this process's thread count (FLOP/s). Used as MFU denominator."""
    torch.set_num_threads(threads)
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    for _ in range(3):
        a @ b
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - t0
    return 2 * n**3 * reps / dt
