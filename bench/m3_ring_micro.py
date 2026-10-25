"""Validate the cost model's ring structure: Gloo all-reduce time vs ring order on the WAN lab.

Two placements with the same per-site counts (2 ranks per site):
  contiguous   tor tor mtl mtl us us   → 3 inter-site ring edges
  interleaved  tor mtl us tor mtl us   → 6 inter-site ring edges (same slowest edge, crossed by more flows)
For each, the all-reduce of the DDP gradient size (from calibration, else --mb) is timed over --iters
iterations per trial. Trials alternate orders. The cost model's prediction (κ = 1) is recorded next to
each measurement.

    python -m bench.m3_ring_micro --trials 5 --out results/m3     (inside the lab container)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from bench.collective_micro import launch
from bench.meta import resolve, write_result
from bench.stats import summarize
from controller import model as M
from netlab.lab import default_lab

ORDERS = {
    "contiguous": ("tor", "tor", "mtl", "mtl", "us", "us"),
    "interleaved": ("tor", "mtl", "us", "tor", "mtl", "us"),
}


def slots(lab, placement):
    used: dict[str, int] = {}
    out = []
    for s in placement:
        used[s] = used.get(s, 0) + 1
        h = f"{s}-c{used[s]}"
        out.append((h, lab.topo.host_ip(h)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--mb", type=float, default=0, help="0 = gradient size from calibration.json")
    ap.add_argument("--out", default="results/m3")
    a = ap.parse_args()
    out = resolve(a.out)
    cal = out / "calibration.json"
    grad_bytes = json.loads(cal.read_text())["data"]["grad_bytes"] if cal.exists() else 19.7e6
    mb = a.mb or grad_bytes / 2**20
    lab = default_lab()
    lab.build()
    lab.start_frr()
    conv = lab.wait_converged()
    net = M.NetState.nominal()
    job = M.JobSpec(n_ranks=6, grad_bytes=mb * 2**20, compute_s=0.0, ckpt_bytes=0.0, ckpt_every=0)
    rows = []
    tmp = out / "raw_ring_micro"
    tmp.mkdir(parents=True, exist_ok=True)
    port = 29650
    try:
        for t in range(a.trials):
            for name in list(ORDERS) if t % 2 == 0 else list(ORDERS)[::-1]:
                port += 1
                res = launch(
                    slots(lab, ORDERS[name]), str(tmp / f"{name}_t{t}.json"), mb=mb, iters=a.iters, port=port
                )
                ts = res["times"]["all_reduce"]
                pred = M.ring_time(net, job, M.make_plan(ORDERS[name], "tor", {}))
                rows.append(
                    {"trial": t, "order": name, "median_s": float(np.median(ts)), "times": ts, "pred_s": pred}
                )
                print(rows[-1]["order"], t, round(rows[-1]["median_s"], 3), "pred", round(pred, 3))
    finally:
        lab.teardown()
    summ = {}
    for name in ORDERS:
        xs = [r["median_s"] for r in rows if r["order"] == name]
        summ[name] = {
            "allreduce_s": summarize(xs).as_dict(),
            "pred_s": next(r["pred_s"] for r in rows if r["order"] == name),
        }
    by_t = {t: {r["order"]: r["median_s"] for r in rows if r["trial"] == t} for t in range(a.trials)}
    ratios = [v["interleaved"] / v["contiguous"] for v in by_t.values()]
    summ["ratio_interleaved_over_contiguous"] = summarize(ratios).as_dict()
    summ["pred_ratio"] = summ["interleaved"]["pred_s"] / summ["contiguous"]["pred_s"]
    write_result(
        out / "ring_micro.json",
        {"summary": summ, "runs": rows, "mb": mb, "bgp_convergence_s": conv},
        config=vars(a),
    )
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
