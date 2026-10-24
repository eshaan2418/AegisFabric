"""Collective micro-benchmark: time all_reduce / all_gather / reduce_scatter of a given size per rank.

Run one process per rank (env RANK/WORLD_SIZE/MASTER_*), e.g. through bench.collective_micro.launch()."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--ops", default="all_reduce")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=dt.timedelta(seconds=120))
    r, w = dist.get_rank(), dist.get_world_size()
    n = int(a.mb * 2**20 / 4) // w * w
    x = torch.ones(n)
    res = {}
    for op in a.ops.split(","):
        ts = []
        for _ in range(a.iters + 1):
            dist.barrier()
            t0 = time.perf_counter()
            if op == "all_reduce":
                dist.all_reduce(x)
            elif op == "all_gather":
                dist.all_gather_into_tensor(x, x[: n // w].clone())
            elif op == "reduce_scatter":
                out = torch.empty(n // w)
                dist.reduce_scatter_tensor(out, x)
            ts.append(time.perf_counter() - t0)
        res[op] = ts[1:]
    if r == 0:
        with open(a.out, "w") as f:
            json.dump({"world": w, "mb": a.mb, "times": res}, f)
    dist.destroy_process_group()


def launch(slots, out: str, mb: float = 16, iters: int = 8, ops: str = "all_reduce", port: int = 29650):
    import subprocess
    import sys

    procs = []
    for rank, (ns, _ip) in enumerate(slots):
        env = dict(
            os.environ,
            RANK=str(rank),
            WORLD_SIZE=str(len(slots)),
            MASTER_ADDR=slots[0][1],
            MASTER_PORT=str(port),
            GLOO_SOCKET_IFNAME="eth0",
            OMP_NUM_THREADS="1",
        )
        cmd = [
            "ip",
            "netns",
            "exec",
            ns,
            sys.executable,
            "-m",
            "bench.collective_micro",
            "--mb",
            str(mb),
            "--iters",
            str(iters),
            "--ops",
            ops,
            "--out",
            out,
        ]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait(timeout=600)
    with open(out) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
