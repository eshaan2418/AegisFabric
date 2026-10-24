"""Figures for M7: latency per implementation (mean ± 95% CI), one panel per (device, shape, dtype).

uv run python -m bench.kernels_figs
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.meta import RESULTS  # noqa: E402

# Colour follows the implementation (entity), fixed order; validated palette (dataviz reference, light mode).
COLORS = {
    "eager_unfused": "#2a78d6",
    "sdpa_mask": "#eb6834",
    "torch_compile": "#1baf7a",
    "jax_jit": "#eda100",
    "metal_fused": "#e87ba4",
}
INK, MUTED = "#1f1f1e", "#6b6a64"


def main() -> None:
    panels = []
    for dev in ("cpu", "mps"):
        p = RESULTS / "m7" / f"perf_{dev}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())["data"]
        keys = sorted({(c["shape"], c["dtype"]) for c in d["cases"]}, key=lambda k: (k[0] != "model", k[1]))
        for shape, dt in keys:
            rows = [c for c in d["cases"] if c["shape"] == shape and c["dtype"] == dt]
            panels.append(
                (
                    f"{'CPU (4 cores, Linux container)' if dev == 'cpu' else 'Apple GPU (MPS)'}"
                    f" — {shape}, {dt}",
                    rows,
                )
            )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7, 1.0 + 1.25 * sum(len(r) for _, r in panels) / 2.2), squeeze=False
    )
    for ax, (title, rows) in zip(axes[:, 0], panels, strict=True):
        names = [r["impl"] for r in rows]
        means = [r["latency_ms"]["mean"] for r in rows]
        half = [(r["latency_ms"]["ci95_hi"] - r["latency_ms"]["ci95_lo"]) / 2 for r in rows]
        y = range(len(rows))
        ax.barh(
            y,
            means,
            xerr=half,
            color=[COLORS[n] for n in names],
            height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2},
        )
        ax.set_yticks(list(y), names, color=INK, fontsize=8)
        ax.invert_yaxis()
        for yi, m in zip(y, means, strict=True):
            ax.text(m * 1.02 + max(means) * 0.01, yi, f"{m:.3g} ms", va="center", fontsize=7, color=INK)
        ax.set_xlim(0, max(means) * 1.25)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=7, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel(
        "latency per call (ms), mean ± 95% CI — dev run, machine shared", fontsize=8, color=MUTED
    )
    fig.tight_layout()
    out = RESULTS / "m7" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "latency.png", dpi=160)
    print("wrote", out / "latency.png")


if __name__ == "__main__":
    main()
