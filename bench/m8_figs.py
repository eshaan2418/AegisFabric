"""M8 federated figure (post-training and DP figures are written by their drivers' `summarize`).

    uv run python -m bench.m8_figs

Reads `results/m8/federated/summary.json`; writes `results/m8/figs/federated_exact.png`.
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


if __name__ == "__main__":
    federated_fig()
