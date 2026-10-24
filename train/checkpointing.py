"""Sliced, asynchronous checkpointing over the network.

Each rank owns ~1/n of the full (placement-independent) training state: leaves are assigned greedily by
size. save(): snapshot (clone this rank's slice, which is the training stall), then a background
thread serialises and uploads `ckpt/<job>/<step>/slice<r>of<n>` to the storage service. restore(): fetch
every slice of the newest step for which all slices exist, then rebuild the full state.

A *backend* has put(key, bytes) and get(key) -> bytes | None. netlab.blobstore.Client is one. The M4
engine client is another (ckpt/).
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import torch


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}\x1f"))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}#{i}\x1f"))
    else:
        out[prefix[:-1]] = d
    return out


def unflatten(flat: dict[str, Any]) -> Any:
    root: dict = {}
    for path, v in flat.items():
        parts = path.split("\x1f")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return _lists(root)


def _lists(d: Any) -> Any:
    if isinstance(d, dict):
        if d and all(k.startswith("#") for k in d):
            return [_lists(d[f"#{i}"]) for i in range(len(d))]
        return {_key(k): _lists(v) for k, v in d.items()}
    return d


def _key(k: str):
    # optimizer state dicts may use integer param ids; FQN-keyed full state dicts use strings
    return int(k) if k.isdigit() else k


def my_slice(flat: dict[str, Any], rank: int, world: int) -> dict[str, Any]:
    """Greedy size-balanced assignment (largest leaf first to the least-loaded rank); deterministic."""

    def size(v: Any) -> int:
        return v.numel() * v.element_size() if isinstance(v, torch.Tensor) else 64

    load = [0] * world
    owner: dict[str, int] = {}
    for k in sorted(flat, key=lambda k: (-size(flat[k]), k)):
        r = min(range(world), key=lambda i: (load[i], i))
        owner[k] = r
        load[r] += size(flat[k])
    return {k: flat[k] for k in sorted(flat) if owner[k] == rank}


class SliceCheckpointer:
    def __init__(self, backend, rank: int, world: int, job: str = "job"):
        self.backend, self.rank, self.world, self.job = backend, rank, world, job
        self._th: threading.Thread | None = None
        self.last_stats: dict = {}
        self.inflight_step: int | None = None

    def key(self, step: int, rank: int | None = None) -> str:
        r = self.rank if rank is None else rank
        return f"ckpt_{self.job}_{step:08d}_slice{r}of{self.world}"

    def busy(self) -> bool:
        return self._th is not None and self._th.is_alive()

    def save(self, step: int, full_state: dict, blocking: bool = False) -> dict:
        """Returns snapshot stats immediately (stall); upload stats land in last_stats when done."""
        wait0 = time.perf_counter()
        self.wait()  # never two flushes in flight from one rank
        waited = time.perf_counter() - wait0
        t0 = time.perf_counter()
        flat = flatten(full_state)
        mine = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in my_slice(flat, self.rank, self.world).items()
        }
        stall = time.perf_counter() - t0
        snap = {"step": step, "stall_s": stall, "wait_prev_s": waited}
        self.inflight_step = step

        def flush() -> None:
            t1 = time.perf_counter()
            buf = io.BytesIO()
            torch.save(mine, buf)
            data = buf.getvalue()
            t2 = time.perf_counter()
            self.backend.put(self.key(step), data)
            self.last_stats = {
                **snap,
                "bytes": len(data),
                "serialize_s": t2 - t1,
                "upload_s": time.perf_counter() - t2,
                "flush_s": time.perf_counter() - t1,
                "done_wall": time.time(),
            }

        self._th = threading.Thread(target=flush, daemon=True)
        self._th.start()
        if blocking:
            self.wait()
        return snap

    def wait(self) -> None:
        if self._th is not None:
            self._th.join()

    def restore(self, step: int) -> dict:
        flat: dict[str, Any] = {}
        for r in range(self.world):
            data = self.backend.get(self.key(step, r))
            if data is None:
                raise FileNotFoundError(self.key(step, r))
            flat.update(torch.load(io.BytesIO(data), weights_only=False))
        return unflatten(flat)
