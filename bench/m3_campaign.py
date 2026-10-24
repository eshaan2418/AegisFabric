"""M3 campaign driver: (scenario × method × seed) WAN runs, resumable (skips existing result files).

    python -m bench.m3_campaign --scenarios all --methods static,ecmp,independent,joint,oracle --seeds 5
    python -m bench.m3_campaign --ablation --scenarios bw_collapse,latency_spike,site_degradation --seeds 5

Each run writes results/m3/raw/<scenario>/<method>/seed<k>.json (see bench/wan_run.py).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

from bench.scenarios import MAIN
from bench.wan_run import RunConfig, run
from controller.telemetry import ALL_SIGNALS

METHODS = ["static", "ecmp", "independent", "joint", "oracle"]


def jobs(a) -> list[RunConfig]:
    scen = MAIN if a.scenarios == "all" else a.scenarios.split(",")
    out = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        for sc in scen:
            if a.ablation:
                for sig in ALL_SIGNALS:
                    sigs = tuple(s for s in ALL_SIGNALS if s != sig)
                    out.append(
                        RunConfig(
                            sc, "joint", seed, steps=a.steps, signals=sigs, variant=f"no-{sig}", out=a.out
                        )
                    )
                out.append(RunConfig(sc, "joint-nomig", seed, steps=a.steps, out=a.out))
            else:
                # methods interleaved per (seed, scenario) block in a seeded random order, so slow drift in
                # host load (the machine is shared) hits all methods alike instead of biasing one
                ms = a.methods.split(",")
                order = np.random.default_rng([seed, zlib.crc32(sc.encode())]).permutation(len(ms))
                for k in order:
                    out.append(RunConfig(sc, ms[k], seed, steps=a.steps, out=a.out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--out", default="results/m3")
    a = ap.parse_args()
    todo = jobs(a)
    log = Path(a.out) / "campaign_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(todo):
        dst = Path(cfg.out) / "raw" / cfg.scenario / cfg.label / f"seed{cfg.seed}.json"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            s = run(cfg)
            status = "ok"
            info = {k: s.get(k) for k in ("total_s", "step_mean_s", "step_p99_s", "migrations", "timed_out")}
        except Exception as e:  # keep the campaign going; failures are logged and re-run on resume
            status = "error"
            info = {"error": repr(e), "tb": traceback.format_exc()[-2000:]}
            from netlab.lab import default_lab

            default_lab().teardown()
        rec = {
            "i": i,
            "n": len(todo),
            "scenario": cfg.scenario,
            "method": cfg.label,
            "seed": cfg.seed,
            "status": status,
            "wall_s": time.time() - t0,
            **info,
        }
        with open(log, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps(rec, default=str)[:400], flush=True)


if __name__ == "__main__":
    main()
