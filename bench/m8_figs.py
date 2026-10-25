"""M8 cross-run figures (per-run bar charts are written by each driver's `summarize`).

    uv run python -m bench.m8_figs

Reads `results/m8/federated/summary.json` and `results/m8/posttrain{,/budget1.6e13}/summary.json`; writes
`results/m8/figs/federated_exact.png` and `results/m8/figs/posttrain_budget.png`.
"""

from __future__ import annotations

from bench import plotstyle
from bench.meta import RESULTS, read_result


def federated_fig() -> None:
    import matplotlib.pyplot as plt

    d = read_result(RESULTS / "m8/federated/summary.json")["data"]["table"]
    names = list(d)
    mean = [d[n]["global_exact"]["mean"] for n in names]
    err = [(d[n]["global_exact"]["ci95_hi"] - d[n]["global_exact"]["ci95_lo"]) / 2 for n in names]
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(names, mean, yerr=err, capsize=4, color=plotstyle.SLOTS[0])
    for i, v in enumerate(mean):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("exact match (mean over 3 site eval sets)")
    ax.set_ylim(0, 1)
    ax.set_title("Federated vs centralized vs local-only (mean ± 95% CI)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    out = RESULTS / "m8" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "federated_exact.png", dpi=150)


def posttrain_budget_fig() -> None:
    import matplotlib.pyplot as plt

    runs = [("posttrain", None), ("posttrain/budget1.6e13", None)]
    pts: dict[str, list[tuple[float, dict]]] = {}
    warm = None
    for sub, _ in runs:
        p = RESULTS / "m8" / sub / "summary.json"
        if not p.exists():
            continue
        d = read_result(p)["data"]["table"]
        warm = d["warm"]["exact"]
        for m, t in d.items():
            if m != "warm":
                pts.setdefault(m, []).append((t["flops"]["mean"], t["exact"]))
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, (m, xs) in enumerate(sorted(pts.items())):
        xs.sort(key=lambda x: x[0])
        ax.errorbar(
            [x for x, _ in xs],
            [s["mean"] for _, s in xs],
            yerr=[(s["ci95_hi"] - s["ci95_lo"]) / 2 for _, s in xs],
            marker="o",
            capsize=3,
            label=m,
            color=plotstyle.SLOTS[i],
        )
    if warm:
        ax.axhline(warm["mean"], color=plotstyle.TEXT2, ls="--", lw=0.8, label="warm start")
    ax.set_xscale("log")
    ax.set_xlabel("post-warm-start analytic FLOPs (RM training not included)")
    ax.set_ylabel("held-out exact match")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "m8" / "figs" / "posttrain_budget.png", dpi=150)


if __name__ == "__main__":
    federated_fig()
    posttrain_budget_fig()
