"""M8 federated fine-tuning: FedAvg / FedProx vs centralized vs local-only, 3 non-IID sites.

  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_federated run --seeds 0 1 2 3 4
  uv run python -m bench.m8_federated summarize

Equal work: every method processes the same total number of local SGD steps (rounds × local_steps × 3 sites)
at the same batch size, starting from the public LM-only base (`bench.m8_privacy.base_model`: pretrained on
public synthetic statements/transcripts, never on tool-use episodes), so all task skill comes from site data.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bench.m8_privacy import base_model
from bench.meta import RESULTS, write_result
from bench.stats import summarize, welch
from data.synth import qa
from posttrain import task
from posttrain.model import DEVICE
from privacy.federated import SITES, central_train, fed_train

OUT = "m8/federated"
HP = dict(rounds=8, local_steps=10, batch=64, lr=2e-3, n_per_site=600, n_eval_per_site=100, fedprox_mu=0.01)


def evaluate_sites(m, evals):
    per = {s: task.evaluate(m, evals[s]) for s in SITES}
    return {
        "global_exact": float(np.mean([per[s]["exact"] for s in SITES])),
        "global_tool_ok": float(np.mean([per[s]["tool_ok"] for s in SITES])),
        "per_site_exact": {s: per[s]["exact"] for s in SITES},
    }


def cmd_run(a):
    base = base_model()
    evals = {s: qa.make_site_split(HP["n_eval_per_site"], 888 + i, s) for i, s in enumerate(SITES)}
    total_steps = HP["rounds"] * HP["local_steps"] * len(SITES)
    for seed in a.seeds:
        torch.manual_seed(seed)
        data = {
            s: qa.make_site_split(HP["n_per_site"], 80_000 + 10 * seed + i, s) for i, s in enumerate(SITES)
        }
        res = {}
        for name, mu in [("fedavg", 0.0), ("fedprox", HP["fedprox_mu"])]:
            t0 = time.perf_counter()
            m, comm = fed_train(base, data, HP["rounds"], HP["local_steps"], HP["batch"], HP["lr"], mu, seed)
            res[name] = {"eval": evaluate_sites(m, evals), "comm": comm, "wall_s": time.perf_counter() - t0}
        t0 = time.perf_counter()
        pooled = [e for s in SITES for e in data[s]]
        m = central_train(base, pooled, total_steps, HP["batch"], HP["lr"], seed)
        res["central"] = {
            "eval": evaluate_sites(m, evals),
            "comm": {
                "raw_data_bytes_pooled": int(sum(len(e.prompt) + len(e.gold_completion) for e in pooled)),
                "note": "centralized training moves every site's raw episodes to one place",
            },
            "wall_s": time.perf_counter() - t0,
        }
        for s in SITES:
            m = central_train(base, data[s], total_steps // len(SITES), HP["batch"], HP["lr"], seed)
            res[f"local_{s}"] = {"eval": evaluate_sites(m, evals)}
        print(seed, {k: round(v["eval"]["global_exact"], 3) for k, v in res.items()}, flush=True)
        write_result(f"{OUT}/seed{seed}.json", res, config={**HP, "device": str(DEVICE)}, seed=seed)


def cmd_summarize(a):
    per = [json.loads(f.read_text())["data"] for f in sorted((RESULTS / OUT).glob("seed*.json"))]
    names = list(per[0])
    table = {
        n: {
            "global_exact": summarize([p[n]["eval"]["global_exact"] for p in per]).as_dict(),
            "per_site_exact": {
                s: summarize([p[n]["eval"]["per_site_exact"][s] for p in per]).as_dict() for s in SITES
            },
        }
        for n in names
    }
    tests = {
        "fedavg_vs_central": welch(
            [p["fedavg"]["eval"]["global_exact"] for p in per],
            [p["central"]["eval"]["global_exact"] for p in per],
        ),
        "fedprox_vs_fedavg": welch(
            [p["fedprox"]["eval"]["global_exact"] for p in per],
            [p["fedavg"]["eval"]["global_exact"] for p in per],
        ),
    }
    comm = per[0]["fedavg"]["comm"]
    write_result(
        f"{OUT}/summary.json", {"table": table, "tests": tests, "comm": comm, "n_seeds": len(per), "hp": HP}
    )
    for n, t in table.items():
        g = t["global_exact"]
        print(
            f"{n:10s} {g['mean']:.3f} [{g['ci95_lo']:.3f},{g['ci95_hi']:.3f}]",
            {s: round(v["mean"], 3) for s, v in t["per_site_exact"].items()},
        )
    print(comm)


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
