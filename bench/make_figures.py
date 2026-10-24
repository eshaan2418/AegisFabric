"""Regenerate every figure from committed result files (no experiments are re-run).

    python -m bench.make_figures

Writes results/<m>/figs/*.png and copies the ones the paper uses into paper/figs/.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from bench import plotstyle
from bench.meta import REPO_ROOT

R = REPO_ROOT / "results"
PAPER_FIGS = REPO_ROOT / "paper" / "figs"


def load(p: Path):
    return json.loads(p.read_text())["data"] if p.exists() else None


def fig_m1() -> list[Path]:
    import matplotlib.pyplot as plt

    d = load(R / "m1" / "baselines.json")
    if not d:
        return []
    s = d["summary"]
    keys = [k for k in s if s[k]["step_mean_s"]["n"]]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ys = [s[k]["step_mean_s"]["mean"] for k in keys]
    es = [(s[k]["step_mean_s"]["ci95_hi"] - s[k]["step_mean_s"]["ci95_lo"]) / 2 for k in keys]
    share = [s[k]["coll_share"]["mean"] for k in keys]
    bars = ax.bar(
        range(len(keys)),
        ys,
        yerr=es,
        color=plotstyle.SLOTS[0],
        width=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": plotstyle.TEXT2},
    )
    for b, sh in zip(bars, share, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{sh * 100:.0f}% comm",
            ha="center",
            va="bottom",
            fontsize=7,
            color=plotstyle.TEXT2,
        )
    ax.set_xticks(range(len(keys)), keys, fontsize=7)
    ax.set_ylabel("mean step time (s)")
    fig.tight_layout()
    out = R / "m1" / "figs" / "m1_step_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return [out]


def fig_m2() -> list[Path]:
    import matplotlib.pyplot as plt

    d = load(R / "m2" / "convergence.json")
    if not d:
        return []
    s = d["summary"]
    kinds = [k for k in ("link_down", "blackhole", "flap", "bgp_reset") if k in s and "outage_s" in s[k]]
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ys = [s[k]["outage_s"]["mean"] for k in kinds]
    es = [(s[k]["outage_s"]["ci95_hi"] - s[k]["outage_s"]["ci95_lo"]) / 2 for k in kinds]
    bars = ax.bar(
        range(len(kinds)),
        ys,
        yerr=es,
        color=plotstyle.SLOTS[0],
        width=0.55,
        error_kw={"elinewidth": 0.8, "ecolor": plotstyle.TEXT2},
    )
    for b, y in zip(bars, ys, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}s", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(kinds)), [k.replace("_", " ") for k in kinds])
    ax.set_ylabel("data-plane outage (s)")
    fig.tight_layout()
    out = R / "m2" / "figs" / "m2_outage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return [out]


def fig_m5() -> list[Path]:
    import matplotlib.pyplot as plt

    d = load(R / "m5" / "rotation.json")
    if not d:
        return []
    s = d["summary"]
    cfgs = [c for c in ("none", "plaintext", "mtls", "mtls-rot") if c in s]
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ys = [s[c]["step_mean_s"]["mean"] for c in cfgs]
    es = [(s[c]["step_mean_s"]["ci95_hi"] - s[c]["step_mean_s"]["ci95_lo"]) / 2 for c in cfgs]
    ax.bar(
        range(len(cfgs)),
        ys,
        yerr=es,
        color=plotstyle.SLOTS[0],
        width=0.55,
        error_kw={"elinewidth": 0.8, "ecolor": plotstyle.TEXT2},
    )
    ax.set_xticks(range(len(cfgs)), cfgs)
    ax.set_ylabel("mean step time (s)")
    fig.tight_layout()
    out = R / "m5" / "figs" / "m5_step_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return [out]


def main() -> int:
    plotstyle.apply()
    made: list[Path] = []
    if (R / "m3" / "raw").exists():
        subprocess.run([sys.executable, "-m", "bench.m3_analyze", "--out", str(R / "m3")], check=True)
        made += sorted((R / "m3" / "figs").glob("*.png"))
    made += fig_m1() + fig_m2() + fig_m5()
    # extension milestones provide their own figure entry points
    for mod in ("bench.ckpt_figures", "bench.media_figures", "bench.kernels_figures", "bench.m8_figures"):
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError:
            continue
        made += list(m.main() or [])
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copy(p, PAPER_FIGS / f"{p.parent.parent.name}_{p.name}")
    print(f"make_figures: {len(made)} figures")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
