"""M8: SFT vs DPO vs GRPO (+GRPO with learned RM) at equal analytic compute on the tool-use task.

  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_posttrain tune --budget 4e12   # LR grid, dev split, seed 100
  uv run python -m bench.m8_posttrain tune-merge
  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_posttrain run --seeds 0 1 2 3 4 --budget 4e12
  uv run python -m bench.m8_posttrain summarize
  # second budget (LRs reused from the 4e12 tuning):
  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_posttrain run --seeds 0 1 2 3 4 --budget 1.6e13 --sub /budget1.6e13
  uv run python -m bench.m8_posttrain summarize --sub /budget1.6e13

Compute budget = analytic FLOPs (2N/token forward incl. sampling & reference/RM passes, 6N/token for
forward+backward) spent *after* the shared warm start. Wall-clock is recorded too, but machine is shared.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from bench.meta import RESULTS, read_result, write_result
from bench.stats import summarize, welch
from data.synth import qa
from posttrain import methods, task
from posttrain.model import DEVICE, ModelConfig

OUT = "m8/posttrain"
WARM = dict(n_sft=2000, pretrain_steps=150, sft_steps=200, batch=64)
EVAL_SEED, DEV_SEED = 777, 998
DEFAULT_LR = {"sft": 1e-3, "dpo": 3e-4, "grpo": 3e-4, "grpo_rm": 3e-4}


def run_method(name: str, model, seed: int, budget: float, lr: float, rm=None):
    if name == "sft":
        return methods.continued_sft(model, seed, budget, lr=lr), {}
    if name == "dpo":
        return methods.dpo(model, seed, budget, lr=lr)
    if name == "grpo":
        return methods.grpo(model, seed, budget, lr=lr)
    if name == "grpo_rm":
        return methods.grpo_rm(model, rm, seed, budget, lr=lr)
    raise ValueError(name)


TUNE_GRID = {"sft": [3e-4, 1e-3], "dpo": [3e-4, 1e-3], "grpo": [1e-4, 3e-4]}
TUNE_SEED = 100


def cmd_tune(a):
    """Run one (method, lr) tuning config, or the whole grid if --method is omitted. Each config writes
    `tune_parts/<method>_<lr>.json`; `tune-merge` picks the best LR per method into `tune.json`."""
    seed = TUNE_SEED
    warm, wc, info = task.warm_start(seed, ModelConfig(), **WARM)
    dev = qa.make_split(256, seed=DEV_SEED)
    configs = [(a.method, a.lr)] if a.method else [(m, lr) for m, lrs in TUNE_GRID.items() for lr in lrs]
    for name, lr in configs:
        m = copy.deepcopy(warm)
        torch.manual_seed(seed)
        c, extra = run_method(name, m, seed, a.budget, lr)
        ev = task.evaluate(m, dev)
        row = {
            "method": name,
            "lr": lr,
            "dev_exact": ev["exact"],
            "dev_tool_ok": ev["tool_ok"],
            "flops": c.flops,
            "wall_s": c.wall_s,
            **extra,
        }
        print(row, flush=True)
        write_result(
            f"{OUT}/tune_parts/{name}_{lr:g}.json",
            {"row": row, "warm_dev": task.evaluate(warm, dev)},
            config={
                "budget": a.budget,
                "warm": WARM,
                "tuning_seed": seed,
                "dev_seed": DEV_SEED,
                "device": str(DEVICE),
            },
            seed=seed,
        )


def cmd_tune_merge(a):
    parts = [read_result(f) for f in sorted((RESULTS / OUT / "tune_parts").glob("*.json"))]
    rows = [p["data"]["row"] for p in parts]
    best = {}
    for name in TUNE_GRID:
        cand = [x for x in rows if x["method"] == name]
        assert len(cand) == len(TUNE_GRID[name]), f"missing tuning parts for {name}"
        best[name] = max(cand, key=lambda x: x["dev_exact"])["lr"]
    best["grpo_rm"] = best["grpo"]
    write_result(
        f"{OUT}/tune.json",
        {"rows": rows, "best_lr": best, "warm_dev": parts[0]["data"]["warm_dev"]},
        config=parts[0]["meta"]["config"],
        seed=TUNE_SEED,
    )
    print(best)


def cmd_run(a):
    tune_p = RESULTS / OUT / "tune.json"
    lrs = read_result(tune_p)["data"]["best_lr"] if tune_p.exists() else DEFAULT_LR
    ev_eps = qa.make_split(a.n_eval, seed=EVAL_SEED)
    for seed in a.seeds:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        warm, wc, info = task.warm_start(seed, ModelConfig(), **WARM)
        res = {"warm": {"eval": task.evaluate(warm, ev_eps), "compute": wc.as_dict(), **info}}
        rm, rm_info = methods.train_reward_model(warm, seed)
        res["reward_model"] = rm_info
        for name in a.methods:
            m = copy.deepcopy(warm)
            torch.manual_seed(seed)
            c, extra = run_method(name, m, seed, a.budget, lrs[name], rm=rm)
            res[name] = {"eval": task.evaluate(m, ev_eps), "compute": c.as_dict(), "lr": lrs[name], **extra}
            print(seed, name, res[name]["eval"]["exact"], f"{c.flops:.2e}", round(c.wall_s), flush=True)
        res["total_wall_s"] = time.perf_counter() - t0
        write_result(
            f"{OUT}{a.sub}/seed{seed}.json",
            res,
            config={
                "budget": a.budget,
                "warm": WARM,
                "lrs": lrs,
                "eval_seed": EVAL_SEED,
                "n_eval": a.n_eval,
                "model": ModelConfig().__dict__,
                "threads": torch.get_num_threads(),
                "device": str(DEVICE),
            },
            seed=seed,
        )


def cmd_summarize(a):
    files = sorted((RESULTS / f"{OUT}{a.sub}").glob("seed*.json"))
    per = [json.loads(f.read_text())["data"] for f in files]
    methods_ = ["warm"] + [m for m in ["sft", "dpo", "grpo", "grpo_rm"] if m in per[0]]
    table = {}
    for m in methods_:
        ex = [p[m]["eval"]["exact"] for p in per]
        to = [p[m]["eval"]["tool_ok"] for p in per]
        fl = [p[m]["compute"]["flops"] for p in per]
        wl = [p[m]["compute"]["wall_s"] for p in per]
        table[m] = {
            "exact": summarize(ex).as_dict(),
            "tool_ok": summarize(to).as_dict(),
            "flops": summarize(fl).as_dict(),
            "wall_s": summarize(wl).as_dict(),
            "by_type": {
                k: float(np.mean([p[m]["eval"]["by_type"].get(k, np.nan) for p in per])) for k in qa.QTYPES
            },
        }
    tests = {}
    for m in methods_[1:]:
        for o in methods_[1:]:
            if m < o:
                tests[f"{m}_vs_{o}"] = welch(
                    [p[m]["eval"]["exact"] for p in per], [p[o]["eval"]["exact"] for p in per]
                )
    rm = {
        "accuracy": summarize([p["reward_model"]["rm_test_accuracy"] for p in per]).as_dict(),
        "auc": summarize([p["reward_model"]["rm_test_auc"] for p in per]).as_dict(),
        "pos_rate": summarize([p["reward_model"]["rm_test_pos_rate"] for p in per]).as_dict(),
    }
    write_result(
        f"{OUT}{a.sub}/summary.json",
        {
            "table": table,
            "tests": tests,
            "reward_model": rm,
            "n_seeds": len(per),
            "files": [f.name for f in files],
        },
    )
    _plot(table, a.sub.strip("/").replace("/", "_"))
    for m, t in table.items():
        e = t["exact"]
        print(
            f"{m:8s} exact {e['mean']:.3f} [{e['ci95_lo']:.3f},{e['ci95_hi']:.3f}] tool_ok {t['tool_ok']['mean']:.3f}"
            f" flops {t['flops']['mean']:.2e} wall {t['wall_s']['mean']:.0f}s"
        )
    print("RM acc", rm["accuracy"]["mean"], "auc", rm["auc"]["mean"])


def _plot(table, suffix: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = RESULTS / "m8" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    names = list(table)
    mean = [table[n]["exact"]["mean"] for n in names]
    err = [(table[n]["exact"]["ci95_hi"] - table[n]["exact"]["ci95_lo"]) / 2 for n in names]
    from bench import plotstyle

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, mean, yerr=err, capsize=4, color=plotstyle.SLOTS[0])
    for i, v in enumerate(mean):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("held-out exact match")
    ax.set_ylim(0, 1)
    ax.set_title("Post-training at equal compute (mean ± 95% CI)")
    fig.tight_layout()
    fig.savefig(figs / f"posttrain_exact{'_' + suffix if suffix else ''}.png", dpi=150)


def main():
    torch.set_num_threads(int(os.environ.get("AEGIS_THREADS", "3")))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "tune-merge", "run", "summarize"])
    ap.add_argument("--method", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--budget", type=float, default=4e12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=["sft", "dpo", "grpo", "grpo_rm"])
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument(
        "--sub", default="", help="output sub-directory, e.g. /budget1.6e13 (LRs still from tune.json)"
    )
    a = ap.parse_args()
    {"tune": cmd_tune, "tune-merge": cmd_tune_merge, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    Path(RESULTS / OUT).mkdir(parents=True, exist_ok=True)
    main()
