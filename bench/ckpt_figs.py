"""Figures for M4 from results/m4/ckpt_bench.json → results/m4/figs/.

uv run python -m bench.ckpt_figs
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.meta import RESULTS, read_result  # noqa: E402

METRICS = [
    ("throughput_MBps", "Durable throughput (MB/s)"),
    ("stall_s", "Training stall per checkpoint (s)"),
    ("write_amp_device", "Device bytes / logical bytes"),
    ("dedup_ratio", "Logical / stored bytes"),
    ("recovery_s", "Recovery to tensors (s)"),
]
ORDER = ["torch_save", "dcp", "aegis_pwrite", "aegis_uring", "aegis_uring_plain", "aegis_uring_noovl"]
COLORS = {
    "torch_save": "#7a7a7a",
    "dcp": "#b0b0b0",
    "aegis_pwrite": "#3b6ea5",
    "aegis_uring": "#1f4e79",
    "aegis_uring_plain": "#8fb3d9",
    "aegis_uring_noovl": "#c9daf0",
}


def main() -> None:
    summ = read_result("m4/ckpt_bench.json")["data"]["summary"]
    out = RESULTS / "m4" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted({k.rsplit("/", 1)[0] for k in summ})  # fs/scenario
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.8 / len(ORDER)
        for j, meth in enumerate(ORDER):
            xs, ys, es = [], [], []
            for i, g in enumerate(groups):
                s = summ.get(f"{g}/{meth}", {}).get(metric)
                if not s or s["n"] == 0:
                    continue
                xs.append(i + (j - (len(ORDER) - 1) / 2) * width)
                ys.append(s["mean"])
                es.append((s["ci95_hi"] - s["ci95_lo"]) / 2)
            if xs:
                ax.bar(xs, ys, width, yerr=es, capsize=2, label=meth, color=COLORS[meth])
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False, ncol=3)
        ax.set_title(f"{label} (mean ± 95% CI)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(METRICS)} figures to {out}")


if __name__ == "__main__":
    main()
