"""Launch a multi-rank training job, optionally placing each rank in a network namespace.

    python -m train.launch --world 4 -- --mode ddp --steps 50
    python -m train.launch --placement tor-h1:10.1.1.11,mtl-h1:10.2.1.11 -- --mode fsdp ...

A placement is an ordered list of (netns, ip). Rank i runs in namespace i, and rank 0's ip is MASTER_ADDR.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Slot:
    netns: str | None
    ip: str
    ifname: str | None = None


def parse_placement(s: str) -> list[Slot]:
    out = []
    for item in s.split(","):
        ns, ip = item.split(":")
        out.append(Slot(ns or None, ip, "eth0"))
    return out


class Job:
    def __init__(
        self,
        slots: list[Slot],
        trainer_args: list[str],
        out: Path,
        port: int = 29500,
        extra_env: dict[str, str] | None = None,
        log_prefix: str = "rank",
    ):
        self.slots, self.args, self.out, self.port = slots, trainer_args, out, port
        self.extra_env = extra_env or {}
        self.procs: list[subprocess.Popen] = []
        self.log_prefix = log_prefix

    def start(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        world = len(self.slots)
        master = self.slots[0].ip
        for r, slot in enumerate(self.slots):
            env = dict(os.environ)
            env.update(
                RANK=str(r),
                WORLD_SIZE=str(world),
                LOCAL_RANK="0",
                MASTER_ADDR=master,
                MASTER_PORT=str(self.port),
                OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"),
                PYTHONUNBUFFERED="1",
            )
            if slot.ifname:
                env["GLOO_SOCKET_IFNAME"] = slot.ifname
            env.update(self.extra_env)
            cmd = [sys.executable, "-m", "train.trainer", *self.args, "--out", str(self.out)]
            if slot.netns:
                cmd = ["ip", "netns", "exec", slot.netns, *cmd]
            log = open(self.out / f"{self.log_prefix}{r}.log", "w")
            self.procs.append(
                subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )

    def wait(self, timeout: float | None = None) -> list[int | None]:
        deadline = time.time() + timeout if timeout else None
        codes: list[int | None] = []
        for p in self.procs:
            remaining = None if deadline is None else max(0.1, deadline - time.time())
            try:
                codes.append(p.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                codes.append(None)
        return codes

    def poll(self) -> list[int | None]:
        return [p.poll() for p in self.procs]

    def kill(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def read_rank_log(out: Path, rank: int = 0) -> list[dict]:
    p = out / f"rank{rank}.jsonl"
    if not p.exists():
        return []
    recs = []
    for line in p.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--placement", default=None)
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--out", type=Path, default=Path("scratch/launch"))
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    slots = parse_placement(a.placement) if a.placement else [Slot(None, "127.0.0.1") for _ in range(a.world)]
    job = Job(slots, rest, a.out, a.port)
    job.start()
    codes = job.wait(a.timeout)
    if any(c != 0 for c in codes):
        job.kill()
        print("rank exit codes:", codes, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
