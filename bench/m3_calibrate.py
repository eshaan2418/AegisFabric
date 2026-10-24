"""Calibrate the controller's cost-model constants on this testbed (run inside the lab container).

compute_s   median step of 6 *concurrent* single-process (local-mode) trainers, the same CPU contention
            as a 6-rank job without any communication
grad_bytes  4 · #params (fp32 gradients all-reduced by DDP)
ckpt_bytes  serialized full training state (params + AdamW moments)
restart_s   launch → first step of a fresh 6-rank job on the WAN lab (process start, data pool, Gloo init)
kappa       measured / predicted ring time for a nominal-network static-routing run (initial value; the
            controller keeps refining it online from collective telemetry)

  python -m bench.m3_calibrate --out results/m3
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bench.meta import write_result
from controller import model as M
from controller.planner import DEFAULT_PLACEMENT
from netlab.lab import default_lab
from train.launch import Job, Slot, read_rank_log
from train.model import PRESETS, TinyVLM


def compute_time(preset: str, batch: int, threads: int, tmp: Path) -> float:
    procs = []
    for k in range(6):
        out = tmp / f"local{k}"
        shutil.rmtree(out, ignore_errors=True)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "train.trainer",
                    "--mode",
                    "local",
                    "--preset",
                    preset,
                    "--batch",
                    str(batch),
                    "--steps",
                    "30",
                    "--eval-every",
                    "0",
                    "--threads",
                    str(threads),
                    "--out",
                    str(out),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    for p in procs:
        p.wait()
    meds = []
    for k in range(6):
        st = [r["step_s"] for r in read_rank_log(tmp / f"local{k}", 0) if r.get("event") == "step"][5:]
        meds.append(float(np.median(st)))
    return float(np.median(meds))


def state_bytes(preset: str) -> tuple[float, float]:
    m = TinyVLM(PRESETS[preset])
    opt = torch.optim.AdamW(m.parameters())
    m.loss(torch.rand(2, 32, 32), torch.randint(1, 255, (2, 128))).backward()
    opt.step()
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
    n = sum(p.numel() for p in m.parameters())
    return 4.0 * n, float(len(buf.getvalue()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/m3")
    a = ap.parse_args()
    tmp = Path("/tmp/aegis_calib")
    tmp.mkdir(parents=True, exist_ok=True)
    comp = [compute_time(a.preset, a.batch, a.threads, tmp) for _ in range(3)]
    grad_b, ckpt_b = state_bytes(a.preset)
    lab = default_lab()
    lab.build()
    lab.start_frr()
    lab.wait_converged()
    restarts, kappas = [], []
    job = M.JobSpec(grad_bytes=grad_b, compute_s=float(np.median(comp)), ckpt_every=0)
    plan = M.make_plan(DEFAULT_PLACEMENT, "mtl", {})
    pred = M.ring_time(M.NetState.nominal(), job, plan)
    for trial in range(3):
        used: dict = {}
        slots = []
        for s in DEFAULT_PLACEMENT:
            used[s] = used.get(s, 0) + 1
            h = f"{s}-c{used[s]}"
            slots.append(Slot(h, lab.topo.host_ip(h), "eth0"))
        out = tmp / f"wan{trial}"
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        j = Job(
            slots,
            [
                "--mode",
                "ddp",
                "--preset",
                a.preset,
                "--batch",
                str(a.batch),
                "--steps",
                "25",
                "--eval-every",
                "0",
                "--threads",
                str(a.threads),
            ],
            out,
            port=29700 + trial,
        )
        j.start()
        j.wait(900)
        recs = read_rank_log(out, 0)
        steps = [r for r in recs if r.get("event") == "step"]
        restarts.append(steps[0]["wall"] - steps[0]["step_s"] - t0)
        meas = float(np.median([r["step_s"] for r in steps[5:]])) - job.compute_s
        kappas.append(meas / pred)
    lab.teardown()
    data = {
        "compute_s": float(np.median(comp)),
        "compute_trials": comp,
        "grad_bytes": grad_b,
        "ckpt_bytes": ckpt_b,
        "restart_s": float(np.median(restarts)),
        "restart_trials": restarts,
        "kappa": float(np.median(kappas)),
        "kappa_trials": kappas,
        "predicted_ring_s_nominal_static": pred,
    }
    write_result(Path(a.out) / "calibration.json", data, config=vars(a))
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
