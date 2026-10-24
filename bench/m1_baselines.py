"""M1: fixed-config baselines for DDP / FSDP2 / HSDP, ≥5 seeds, on two networks:
  loopback  all 6 ranks in one namespace (no WAN), the compute-bound reference
  wan       2 ranks per site on the emulated WAN, default (static) routing, nominal links

Each run trains to --target-loss (eval loss on held-out synthetic data, evaluated every 20 steps) or
--steps. Records time-to-target, step-time stats, per-collective time share, tokens/s, MFU, RSS.

    python -m bench.m1_baselines --seeds 5 --out results/m1
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from bench.meta import write_result
from bench.stats import summarize
from controller.planner import DEFAULT_PLACEMENT
from train.launch import Job, Slot, read_rank_log

MODES = {"ddp": [], "fsdp": [], "hsdp": ["--sites", "3"]}


def slots(net: str, lab) -> list[Slot]:
    if net == "loopback":
        return [Slot(None, "127.0.0.1", "lo") for _ in range(6)]
    used: dict = {}
    out = []
    for s in DEFAULT_PLACEMENT:
        used[s] = used.get(s, 0) + 1
        h = f"{s}-c{used[s]}"
        out.append(Slot(h, lab.topo.host_ip(h), "eth0"))
    return out


def one(net: str, mode: str, seed: int, args, lab, tmp: Path) -> dict:
    out = tmp / f"{net}_{mode}_{seed}"
    shutil.rmtree(out, ignore_errors=True)
    targs = [
        "--mode",
        mode,
        *MODES[mode],
        "--preset",
        args.preset,
        "--batch",
        str(args.batch),
        "--steps",
        str(args.steps),
        "--eval-every",
        "20",
        "--threads",
        str(args.threads),
        "--seed",
        str(seed),
        "--data-seed",
        str(1234 + seed),
        "--target-loss",
        str(args.target_loss),
    ]
    t0 = time.time()
    j = Job(slots(net, lab), targs, out, port=30000 + seed * 10 + list(MODES).index(mode))
    j.start()
    codes = j.wait(3600)
    recs = read_rank_log(out, 0)
    steps = [r for r in recs if r.get("event") == "step"]
    start = next(r for r in recs if r.get("event") == "start")
    end = next((r for r in recs if r.get("event") == "end"), {})
    st = np.array([r["step_s"] for r in steps[3:]])
    coll = np.array([sum(v["time_s"] for v in r["coll"].values()) for r in steps[3:]])
    res = {
        "net": net,
        "mode": mode,
        "seed": seed,
        "exit": codes,
        "params": start["params"],
        "steps": len(steps),
        "reached_target_s": end.get("reached_target_s"),
        "final_eval": end.get("eval_loss"),
        "stop_reason": end.get("stop_reason"),
        "launch_to_end_s": time.time() - t0,
        "step_mean_s": float(st.mean()),
        "step_p50_s": float(np.median(st)),
        "step_p99_s": float(np.percentile(st, 99)),
        "coll_time_mean_s": float(coll.mean()),
        "coll_share": float(coll.mean() / st.mean()),
        "tok_s_mean": float(np.mean([r["tok_s"] for r in steps[3:]])),
        "mfu_mean": float(np.mean([r["mfu"] for r in steps[3:]])),
        "rss_mb_max": float(max(r["rss_mb"] for r in steps)),
        "peak_flops_per_rank": start["peak_flops_per_rank"],
        "coll_breakdown": {
            k: float(np.mean([r["coll"].get(k, {}).get("time_s", 0) for r in steps[3:]]))
            for k in sorted({k for r in steps for k in r["coll"]})
        },
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--nets", default="loopback,wan")
    ap.add_argument("--modes", default="ddp,fsdp,hsdp")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--target-loss", type=float, default=1.6)
    ap.add_argument("--out", default="results/m1")
    a = ap.parse_args()
    tmp = Path("/tmp/aegis_m1")
    tmp.mkdir(parents=True, exist_ok=True)
    lab = None
    rows = []
    for net in a.nets.split(","):
        if net == "wan":
            from netlab.lab import default_lab

            lab = default_lab()
            lab.build()
            lab.start_frr()
            lab.wait_converged()
        for mode in a.modes.split(","):
            for seed in range(a.seeds):
                r = one(net, mode, seed, a, lab, tmp)
                print(
                    {k: r[k] for k in ("net", "mode", "seed", "reached_target_s", "step_p50_s", "coll_share")}
                )
                rows.append(r)
                write_result(
                    Path(a.out) / "raw" / f"{net}_{mode}_seed{seed}.json", r, config=vars(a), seed=seed
                )
        if lab is not None:
            lab.teardown()
            lab = None
    summary = {}
    for net in a.nets.split(","):
        for mode in a.modes.split(","):
            rs = [r for r in rows if r["net"] == net and r["mode"] == mode]
            summary[f"{net}/{mode}"] = {
                k: summarize([r[k] for r in rs if r[k] is not None], with_p99=False).as_dict()
                for k in (
                    "reached_target_s",
                    "step_mean_s",
                    "step_p99_s",
                    "coll_share",
                    "tok_s_mean",
                    "mfu_mean",
                    "rss_mb_max",
                )
            }
            summary[f"{net}/{mode}"]["n_reached"] = sum(r["reached_target_s"] is not None for r in rs)
    write_result(Path(a.out) / "baselines.json", {"summary": summary, "runs": rows}, config=vars(a))


if __name__ == "__main__":
    main()
