"""Figures + tables for M4 (and the M5 envelope overhead) -> results/m4/figs/*.png and figs/TABLES.md.

    uv run python -m bench.m4_figs

Inputs (each skipped if absent): results/m4/bench.json, results/m4/dedup.json,
results/m4/crash_{kill9,torn,bitflip}.json, results/m5/envelope_overhead.json.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from bench import plotstyle
from bench.meta import RESULTS, read_result

METRICS = [
    ("throughput_MBps", "Durable throughput (MB/s)"),
    ("stall_s", "Training stall per checkpoint (s)"),
    ("write_amp_device", "Device bytes / logical bytes"),
    ("dedup_ratio", "Logical / uploaded bytes"),
    ("recovery_s", "Recovery to tensors (s)"),
]
ORDER = ["torch_save", "dcp", "aegis_pwrite", "aegis_uring", "aegis_uring_plain", "aegis_uring_sync"]
COLORS = dict(zip(ORDER, plotstyle.SLOTS, strict=False))


def _half(s: dict) -> float:
    return (s["ci95_hi"] - s["ci95_lo"]) / 2 if s["n"] > 1 else 0.0


def bench_figs(out, tables: list[str]) -> None:
    p = RESULTS / "m4" / "bench.json"
    if not p.exists():
        return
    summ = read_result(p)["data"]["summary"]
    groups = sorted({k.rsplit("/", 1)[0] for k in summ})
    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        width = 0.8 / len(ORDER)
        tables.append(f"\n### {label} (`{metric}`, mean ± 95% CI half-width)\n")
        tables.append("| fs/scenario | " + " | ".join(ORDER) + " |")
        tables.append("|---" * (len(ORDER) + 1) + "|")
        rows = {g: [] for g in groups}
        for j, meth in enumerate(ORDER):
            for i, g in enumerate(groups):
                s = summ.get(f"{g}/{meth}", {}).get(metric)
                if not s or s["n"] == 0:
                    rows[g].append("–")
                    continue
                x = i + (j - (len(ORDER) - 1) / 2) * width
                ax.bar(
                    x,
                    s["mean"],
                    width,
                    yerr=_half(s),
                    capsize=2,
                    color=COLORS[meth],
                    label=meth if i == 0 else None,
                )
                ax.text(
                    x,
                    s["mean"] + _half(s),
                    f"{s['mean']:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
                rows[g].append(f"{s['mean']:.3g} ± {_half(s):.2g} (n={s['n']})")
        for g in groups:
            tables.append(f"| {g} | " + " | ".join(rows[g]) + " |")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylabel(label)
        if metric in ("stall_s", "recovery_s"):
            ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.set_title(f"{label} (mean ± 95% CI)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"bench_{metric}.png", dpi=150)
        plt.close(fig)


def dedup_fig(out, tables: list[str]) -> None:
    p = RESULTS / "m4" / "dedup.json"
    if not p.exists():
        return
    summ = read_result(p)["data"]["summary"]
    series: dict[str, list] = {}
    for k, v in summ.items():
        scen, scheme, size = k.split("/")
        series.setdefault(f"{scen}/{scheme}", []).append((int(size), v["dedup_ratio"]))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    tables.append("\n### Dedup ratio vs chunk size (`results/m4/dedup.json`)\n")
    tables.append("| scenario/scheme | chunk | dedup ratio (mean ± CI) |")
    tables.append("|---|---|---|")
    for i, (name, pts) in enumerate(sorted(series.items())):
        pts.sort()
        xs = [p[0] / 1024 for p in pts]
        ys = [p[1]["mean"] for p in pts]
        es = [_half(p[1]) for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=name, color=plotstyle.SLOTS[i], capsize=2)
        for (sz, s), y in zip(pts, ys, strict=True):
            tables.append(f"| {name} | {sz // 1024} KiB | {y:.3f} ± {_half(s):.2g} (n={s['n']}) |")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("chunk size (KiB)")
    ax.set_ylabel("logical / unique bytes")
    ax.legend(fontsize=7)
    ax.set_title("Dedup across successive training checkpoints", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "dedup_chunksize.png", dpi=150)
    plt.close(fig)


def crash_fig(out, tables: list[str]) -> None:
    rows = []
    for kind in ("kill9", "torn", "bitflip"):
        p = RESULTS / "m4" / f"crash_{kind}.json"
        if p.exists():
            rows.append((kind, read_result(p)["data"]["outcomes"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 2.8))
    tables.append("\n### Crash campaign outcomes (`results/m4/crash_*.json`)\n")
    tables.append("| campaign | ok | detected | undetected |")
    tables.append("|---|---|---|---|")
    for i, (kind, o) in enumerate(rows):
        left = 0
        for j, k in enumerate(("ok", "detected", "undetected")):
            ax.barh(i, o[k], left=left, color=plotstyle.SLOTS[[2, 3, 7][j]], label=k if i == 0 else None)
            if o[k]:
                ax.text(left + o[k] / 2, i, str(o[k]), ha="center", va="center", fontsize=7)
            left += o[k]
        tables.append(f"| {kind} | {o['ok']} | {o['detected']} | {o['undetected']} |")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("trials")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.set_title("Recovery outcomes (undetected must be 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "crash_outcomes.png", dpi=150)
    plt.close(fig)


def envelope_fig(out, tables: list[str]) -> None:
    p = RESULTS / "m5" / "envelope_overhead.json"
    if not p.exists():
        return
    d = read_result(p)["data"]
    summ = d["summary"]
    keys = [("cold_s", "cold restore"), ("warm_s", "warm restore"), ("cold_verify_s", "verify (cold)")]
    fig, ax = plt.subplots(figsize=(5, 3))
    tables.append("\n### P2 envelope overhead (`results/m5/envelope_overhead.json`)\n")
    tables.append("| metric | plaintext | P2 | ratio |")
    tables.append("|---|---|---|---|")
    for j, mode in enumerate(("plain", "p2")):
        xs = [i + (j - 0.5) * 0.38 for i in range(len(keys))]
        ys = [summ[mode][k]["mean"] for k, _ in keys]
        es = [_half(summ[mode][k]) for k, _ in keys]
        ax.bar(xs, ys, 0.38, yerr=es, capsize=2, color=plotstyle.SLOTS[j], label=mode)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6)
    for k, lab in keys:
        a, b = summ["plain"][k], summ["p2"][k]
        tables.append(
            f"| {lab} | {a['mean']:.3f} ± {_half(a):.2g} s | {b['mean']:.3f} ± {_half(b):.2g} s | "
            f"{d['overhead'][k]['p2_over_plain']:.2f}× |"
        )
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
    ax.set_ylabel("seconds")
    ax.legend(fontsize=7)
    ax.set_title(f"Restore of a {d['state_bytes'] / 1e6:.0f} MB checkpoint", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "envelope_overhead.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plotstyle.apply()
    out = RESULTS / "m4" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    tables = ["# M4/M5 figure tables (generated by `python -m bench.m4_figs`)"]
    bench_figs(out, tables)
    dedup_fig(out, tables)
    crash_fig(out, tables)
    envelope_fig(out, tables)
    (out / "TABLES.md").write_text("\n".join(tables) + "\n")
    print(f"wrote figures and TABLES.md to {out}")


if __name__ == "__main__":
    main()
