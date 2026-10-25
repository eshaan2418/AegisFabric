"""M8 privacy: DP-SGD (Opacus) on the private fine-tuning step; MIA (loss-threshold + offline LiRA) on canaries.

  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_privacy shadows --eps inf 16 4 1   # K OUT shadows per ε
  AEGIS_DEVICE=mps nice -n 19 uv run python -m bench.m8_privacy run --eps inf 16 4 1 --seeds 0 1 2 3 4
  uv run python -m bench.m8_privacy summarize

Fixed public task model (LM pretrain + public SFT, seed 0; `task_base_model`). The seed varies: member/non-member split, private task data,
data order and DP noise.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bench.meta import RESULTS, read_result, write_result
from bench.stats import summarize
from data.synth import qa
from posttrain import task
from posttrain.model import DEVICE, ModelConfig
from privacy import dp_mia as D

OUT = "m8/privacy"
HP = dict(epochs=6, batch=128, lr=1e-3, delta=1e-5, max_grad_norm=1.0)
N_TASK, N_POOL, POOL_SEED, K_SHADOW, N_EVAL = 1200, 200, 4242, 3, 300
_BASE = {}


def base_model():
    """Public base model, cached on disk so shadow and target runs share bit-identical weights."""
    if "m" not in _BASE:
        from pathlib import Path

        from posttrain.model import DEVICE, TinyLM

        p = Path("data/generated/m8_privacy_base.pt")
        if p.exists():
            m = TinyLM(ModelConfig()).to(DEVICE)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
        else:
            m = D.pretrain_base(0, ModelConfig(), steps=150, batch=64)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, p)
        _BASE["m"] = m
    return _BASE["m"]


def task_base_model():
    """Public *task* model for the private step: LM pretraining + SFT on public gold episodes (seed 0).
    The private step then fine-tunes it on private records (+canaries). Cached on disk."""
    if "t" not in _BASE:
        from pathlib import Path

        from posttrain.model import DEVICE, TinyLM

        p = Path("data/generated/m8_privacy_taskbase.pt")
        if p.exists():
            m = TinyLM(ModelConfig()).to(DEVICE)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
        else:
            m, _, _ = task.warm_start(
                0, ModelConfig(), n_sft=2000, pretrain_steps=150, sft_steps=200, batch=64
            )
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in m.state_dict().items()}, p)
        _BASE["t"] = m
    return _BASE["t"]


def eps_val(s: str) -> float | None:
    return None if s == "inf" else float(s)


def cmd_shadows(a):
    base = task_base_model()
    pool = D.make_canaries(N_POOL, POOL_SEED)
    for e in a.eps:
        losses = []
        for k in range(K_SHADOW):
            ds = D.build_dataset(qa.make_split(N_TASK, seed=70_000 + k), [], base.cfg.max_len)
            m, info = D.finetune(base, ds, 900 + k, epsilon=eps_val(e), **HP)
            losses.append(D.canary_losses(m, pool).tolist())
            print("shadow", e, k, info.get("epsilon_spent_rdp"), flush=True)
        write_result(
            f"{OUT}/shadows_eps{e}.json",
            {"losses": losses},
            config={"hp": HP, "n_task": N_TASK, "pool_seed": POOL_SEED, "eps": e, "K": K_SHADOW},
        )


def cmd_run(a):
    base = task_base_model()
    pool = D.make_canaries(N_POOL, POOL_SEED)
    ev = qa.make_split(N_EVAL, seed=777)
    if not (
        RESULTS / OUT / "base_eval.json"
    ).exists():  # utility of the public task model before the private step
        write_result(f"{OUT}/base_eval.json", {"utility": task.evaluate(base, ev)}, config={"n_eval": N_EVAL})
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        member = np.zeros(N_POOL, bool)
        member[rng.permutation(N_POOL)[: N_POOL // 2]] = True
        task_eps = qa.make_split(N_TASK, seed=60_000 + seed)
        ds = D.build_dataset(task_eps, [c for c, m in zip(pool, member, strict=True) if m], base.cfg.max_len)
        for e in a.eps:
            t0 = time.perf_counter()
            m, info = D.finetune(base, ds, seed, epsilon=eps_val(e), **HP)
            wall = time.perf_counter() - t0
            loss = D.canary_losses(m, pool)
            sh = np.array(read_result(f"{OUT}/shadows_eps{e}.json")["data"]["losses"])
            res = {
                "dp": info,
                "train_wall_s": wall,
                "utility": task.evaluate(m, ev),
                "mia_loss_threshold": D.attack_metrics(-loss, member, seed=seed),
                "mia_lira_offline": D.attack_metrics(D.lira_offline_scores(loss, sh), member, seed=seed),
                "mia_lira_offline_globalvar": D.attack_metrics(
                    D.lira_offline_scores(loss, sh, global_var=True), member, seed=seed
                ),
                "mia_shadow_calibrated": D.attack_metrics(sh.mean(0) - loss, member, seed=seed),
                "canary_losses": loss.tolist(),
                "is_member": member.astype(int).tolist(),
                "canary_loss_member_mean": float(loss[member].mean()),
                "canary_loss_nonmember_mean": float(loss[~member].mean()),
            }
            print(
                seed,
                e,
                res["utility"]["exact"],
                res["mia_loss_threshold"]["auc"],
                res["mia_lira_offline"]["auc"],
                info.get("epsilon_spent_rdp"),
                flush=True,
            )
            write_result(
                f"{OUT}/seed{seed}_eps{e}.json",
                res,
                config={
                    "hp": HP,
                    "n_task": N_TASK,
                    "n_pool": N_POOL,
                    "pool_seed": POOL_SEED,
                    "members": int(member.sum()),
                    "device": str(DEVICE),
                    "eps": e,
                },
                seed=seed,
            )


def cmd_summarize(a):
    rows = {}
    for f in sorted((RESULTS / OUT).glob("seed*_eps*.json")):
        d = json.loads(f.read_text())["data"]
        e = f.stem.split("_eps")[1]
        rows.setdefault(e, []).append(d)
    table = {}
    for e, ds in rows.items():
        table[e] = {
            "n_seeds": len(ds),
            "epsilon_rdp": summarize([d["dp"].get("epsilon_spent_rdp", np.inf) for d in ds]).as_dict()
            if e != "inf"
            else None,
            "epsilon_prv": summarize([d["dp"].get("epsilon_spent_prv", np.nan) for d in ds]).as_dict()
            if e != "inf"
            else None,
            "noise_multiplier": ds[0]["dp"].get("noise_multiplier"),
            "steps": ds[0]["dp"]["steps"],
            "sample_rate": ds[0]["dp"]["sample_rate"],
            "utility_exact": summarize([d["utility"]["exact"] for d in ds]).as_dict(),
            "utility_tool_ok": summarize([d["utility"]["tool_ok"] for d in ds]).as_dict(),
            "mia_loss_auc": summarize([d["mia_loss_threshold"]["auc"] for d in ds]).as_dict(),
            "mia_lira_auc": summarize([d["mia_lira_offline"]["auc"] for d in ds]).as_dict(),
            "mia_lira_globalvar_auc": summarize(
                [d["mia_lira_offline_globalvar"]["auc"] for d in ds]
            ).as_dict(),
            "mia_calibrated_auc": summarize([d["mia_shadow_calibrated"]["auc"] for d in ds]).as_dict(),
            "lira_tpr@5%fpr": summarize([d["mia_lira_offline"]["tpr@5%fpr"] for d in ds]).as_dict(),
            "train_wall_s": summarize([d["train_wall_s"] for d in ds]).as_dict(),
        }
    write_result(f"{OUT}/summary.json", {"table": table, "hp": HP, "delta": HP["delta"]})
    _plot(table)
    for e, t in table.items():
        print(
            e,
            "util",
            round(t["utility_exact"]["mean"], 3),
            "lossAUC",
            round(t["mia_loss_auc"]["mean"], 3),
            "liraAUC",
            round(t["mia_lira_auc"]["mean"], 3),
            "eps_rdp",
            t["epsilon_rdp"] and round(t["epsilon_rdp"]["mean"], 2),
        )


def _plot(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = RESULTS / "m8" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    order = sorted(table, key=lambda e: float("inf") if e == "inf" else float(e))
    from bench import plotstyle

    plotstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for key, lab in [("mia_loss_auc", "loss-threshold"), ("mia_lira_auc", "offline LiRA")]:
        ax[0].errorbar(
            order,
            [table[e][key]["mean"] for e in order],
            yerr=[(table[e][key]["ci95_hi"] - table[e][key]["ci95_lo"]) / 2 for e in order],
            marker="o",
            capsize=3,
            label=lab,
            color=plotstyle.SLOTS[0 if key == "mia_loss_auc" else 1],
        )
    ax[0].axhline(0.5, color=plotstyle.TEXT2, ls="--", lw=0.8)
    ax[0].set_xlabel("target ε (δ=1e-5); inf = non-private")
    ax[0].set_ylabel("MIA AUC")
    ax[0].legend()
    ax[1].errorbar(
        order,
        [table[e]["utility_exact"]["mean"] for e in order],
        yerr=[
            (table[e]["utility_exact"]["ci95_hi"] - table[e]["utility_exact"]["ci95_lo"]) / 2 for e in order
        ],
        marker="o",
        capsize=3,
        color=plotstyle.SLOTS[2],
    )
    ax[1].set_xlabel("target ε")
    ax[1].set_ylabel("task exact match")
    fig.tight_layout()
    fig.savefig(figs / "dp_mia_utility.png", dpi=150)


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shadows", "run", "summarize"])
    ap.add_argument("--eps", nargs="+", default=["inf", "16", "4", "1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    {"shadows": cmd_shadows, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
