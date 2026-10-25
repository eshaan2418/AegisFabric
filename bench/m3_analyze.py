"""Aggregate M3 runs → results/m3/summary.json, results/m3/TABLES.md, results/m3/figs/*.png

    python -m bench.m3_analyze --out results/m3 [--target-loss 0.9]

Statistics: per (scenario, method) mean ± 95% CI over seeds; joint vs each baseline: Welch t-test and
Mann-Whitney U, Holm-corrected across all (scenario × baseline) comparisons of the same metric.
Recovery time per failure event = inject → first of 3 consecutive steps ≤ 1.2× the pre-failure median step;
an event that never recovers within the run is *censored* and counted separately (not averaged).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench import plotstyle
from bench.meta import resolve, write_result
from bench.stats import bootstrap_ci, holm, summarize, welch

BASELINES = ["static", "ecmp", "independent", "oracle"]
METRICS = ["total_s", "ttq_s", "step_mean_s", "step_p99_s", "step_std_s"]


def load(out: Path) -> dict:
    runs: dict = defaultdict(lambda: defaultdict(list))
    for p in sorted((out / "raw").glob("*/*/seed*.json")):
        d = json.loads(p.read_text())
        runs[p.parent.parent.name][p.parent.name].append(d["data"]["summary"] | {"seed": d["meta"]["seed"]})
    return runs


TEST_METRICS = ["total_s", "ttq_s", "step_mean_s", "step_p99_s", "step_std_s", "excess_mean_s"]


def _paired(a: list[dict], b: list[dict], metric: str) -> tuple[list[float], list[float]]:
    """Seed-matched values (same seed ⇒ same failure times, pair and background schedule)."""
    bs = {r["seed"]: r.get(metric) for r in b}
    xa, xb = [], []
    for r in a:
        va, vb = r.get(metric), bs.get(r["seed"])
        if va is not None and vb is not None:
            xa.append(float(va))
            xb.append(float(vb))
    return xa, xb


def _test(a: list[dict], b: list[dict], metric: str) -> dict | None:
    from scipy import stats as st

    xa = [float(r[metric]) for r in a if r.get(metric) is not None]
    xb = [float(r[metric]) for r in b if r.get(metric) is not None]
    if len(xa) < 2 or len(xb) < 2:
        return None
    out = welch(xa, xb)
    pa, pb = _paired(a, b, metric)
    out.update({"n_a": len(xa), "n_b": len(xb), "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb))})
    if len(pa) >= 2:
        d = np.asarray(pa) - np.asarray(pb)
        out["n_paired"] = len(pa)
        out["mean_diff_paired"] = float(d.mean())
        out["p_paired"] = float(st.ttest_rel(pa, pb).pvalue) if np.any(d != d[0]) else float("nan")
        rel = (np.asarray(pb) - np.asarray(pa)) / np.asarray(pb)  # >0: `a` (joint) better
        out["rel_improvement_paired"] = float(rel.mean())
    return out


def compare(runs: dict) -> dict:
    """joint vs each baseline (family 'baselines') and joint vs each ablation variant (family 'ablations').
    Holm correction per (family, metric) across all scenarios × comparators, applied to the paired p-value
    (seeds are matched) when available, else Welch."""
    out: dict = {"baselines": {}, "ablations": {}}
    for fam in out:
        for metric in TEST_METRICS:
            raw, pv = {}, {}
            for sc, by_m in runs.items():
                if "joint" not in by_m:
                    continue
                comps = BASELINES if fam == "baselines" else sorted(k for k in by_m if k.startswith("joint-"))
                for b in comps:
                    if b not in by_m:
                        continue
                    t = _test(by_m["joint"], by_m[b], metric)
                    if t is None:
                        continue
                    key = f"{sc}|{b}"
                    raw[key] = t
                    p = t.get("p_paired")
                    pv[key] = p if p is not None and np.isfinite(p) else t["p_welch"]
            adj = holm(pv) if pv else {}
            out[fam][metric] = {k: {**raw[k], "p_holm": adj[k]} for k in raw}
    return out


def pooled_effects(runs: dict) -> dict:
    """Across scenarios: mean seed-paired relative improvement of joint over each comparator, with a
    bootstrap 95% CI over (scenario, seed) pairs and a Wilcoxon signed-rank test."""
    from scipy import stats as st

    res: dict = {}
    comps = set(BASELINES) | {k for by_m in runs.values() for k in by_m if k.startswith("joint-")}
    for b in sorted(comps):
        res[b] = {}
        for metric in TEST_METRICS:
            rel = []
            for by_m in runs.values():
                if "joint" in by_m and b in by_m:
                    pa, pb = _paired(by_m["joint"], by_m[b], metric)
                    rel += [(y - x) / y for x, y in zip(pa, pb, strict=True) if y]
            if len(rel) < 3:
                continue
            lo, hi = bootstrap_ci(rel)
            try:
                p = float(st.wilcoxon(rel).pvalue)
            except ValueError:
                p = float("nan")
            res[b][metric] = {
                "n_pairs": len(rel),
                "mean_rel_improvement": float(np.mean(rel)),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "median": float(np.median(rel)),
                "p_wilcoxon": p,
                "frac_joint_better": float(np.mean(np.asarray(rel) > 0)),
            }
    return res


def ttq(s: dict, target: float) -> float | None:
    for t, loss, _step in s.get("eval", []):
        if loss <= target:
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m3")
    ap.add_argument("--target-loss", type=float, default=0.9)
    a = ap.parse_args()
    out = resolve(a.out)
    runs = load(out)
    table: dict = {}
    for sc, by_m in runs.items():
        table[sc] = {}
        for m, rs in by_m.items():
            for r in rs:
                r["ttq_s"] = ttq(r, a.target_loss)
            row = {k: summarize([r[k] for r in rs if r.get(k) is not None]).as_dict() for k in METRICS}
            rec = [e for r in rs for e in r["recovery"]]
            got = [e["recovery_s"] for e in rec if e["recovery_s"] is not None]
            row["recovery_s"] = summarize(got).as_dict() if got else None
            row["recovery_censored"] = sum(e["recovery_s"] is None for e in rec)
            row["recovery_events"] = len(rec)
            row["excess_s_60"] = summarize([e["excess_s_60"] for e in rec]).as_dict() if rec else None
            row["migrations_mean"] = float(np.mean([r["migrations"] for r in rs]))
            row["n"] = len(rs)
            row["timed_out"] = sum(bool(r.get("timed_out")) for r in rs)
            row["crashes"] = sum(int(r.get("n_crashes", 0)) for r in rs)
            table[sc][m] = row
    for by_m in runs.values():
        for rs in by_m.values():
            for r in rs:
                ev = r["recovery"]
                r["excess_mean_s"] = float(np.mean([e["excess_s_60"] for e in ev])) if ev else None
    tests = compare(runs)
    pooled = pooled_effects(runs)
    write_result(
        out / "summary.json",
        {"table": table, "tests": tests, "pooled": pooled, "target_loss": a.target_loss},
        config=vars(a),
    )
    write_tables(out, table, tests, pooled)
    figures(out, runs, table)


def fmt(s: dict | None, d: int = 3) -> str:
    if not s or s.get("n", 0) == 0:
        return "–"
    half = (s["ci95_hi"] - s["ci95_lo"]) / 2 if s["n"] > 1 else float("nan")
    return f"{s['mean']:.{d}g} ± {half:.2g}"


def write_tables(out: Path, table: dict, tests: dict, pooled: dict) -> None:
    lines = [
        "# M3 tables (generated by `python -m bench.m3_analyze`; source: results/m3/summary.json)",
        "",
        "Mean ± 95% CI half-width over seeds. Recovery: mean over recovered failure events; censored = never "
        "recovered within the run.",
        "",
    ]
    for sc in sorted(table):
        lines += [
            f"## {sc}",
            "",
            "| method | n | total (s) | time-to-quality (s) | step mean (s) | step p99 (s) | "
            "step std (s) | recovery (s) | censored/events | migrations | crashes | timed out |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in ["static", "ecmp", "independent", "joint", "oracle"] + sorted(
            k for k in table[sc] if k not in ("static", "ecmp", "independent", "joint", "oracle")
        ):
            if m not in table[sc]:
                continue
            r = table[sc][m]
            lines.append(
                f"| {m} | {r['n']} | {fmt(r['total_s'])} | {fmt(r['ttq_s'])} | {fmt(r['step_mean_s'])} | "
                f"{fmt(r['step_p99_s'])} | {fmt(r['step_std_s'])} | {fmt(r['recovery_s'])} | "
                f"{r['recovery_censored']}/{r['recovery_events']} | {r['migrations_mean']:.1f} | "
                f"{r['crashes']} | {r['timed_out']} |"
            )
        lines.append("")
    lines += [
        "## Joint vs comparators (seed-paired t-test; Holm-corrected per family × metric)",
        "",
        "rel. impr. = mean over seeds of (comparator − joint)/comparator; > 0 means joint is better.",
        "",
        "| family | metric | scenario | comparator | joint | comparator | rel. impr. | p (paired) | p (Welch) "
        "| p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam, by_metric in tests.items():
        for metric, t in by_metric.items():
            for k in sorted(t):
                sc, b = k.split("|")
                v = t[k]
                lines.append(
                    f"| {fam} | {metric} | {sc} | {b} | {v['mean_a']:.3g} | {v['mean_b']:.3g} | "
                    f"{v.get('rel_improvement_paired', float('nan')) * 100:+.1f}% | "
                    f"{v.get('p_paired', float('nan')):.3g} | {v['p_welch']:.3g} | {v['p_holm']:.3g} |"
                )
    lines += [
        "",
        "## Pooled over scenarios (seed-paired relative improvement of joint; bootstrap 95% CI; Wilcoxon)",
        "",
        "| comparator | metric | pairs | mean rel. impr. | 95% CI | joint better (frac) | p (Wilcoxon) |",
        "|---|---|---|---|---|---|---|",
    ]
    for b, by_metric in pooled.items():
        for metric, v in by_metric.items():
            lines.append(
                f"| {b} | {metric} | {v['n_pairs']} | {v['mean_rel_improvement'] * 100:+.1f}% | "
                f"[{v['ci95_lo'] * 100:+.1f}%, {v['ci95_hi'] * 100:+.1f}%] | {v['frac_joint_better']:.2f} | "
                f"{v['p_wilcoxon']:.3g} |"
            )
    (out / "TABLES.md").write_text("\n".join(lines) + "\n")


def figures(out: Path, runs: dict, table: dict) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["static", "ecmp", "independent", "joint", "oracle"]]
    scen = sorted(table)
    for metric, label in (("total_s", "training time to 160 steps (s)"), ("step_p99_s", "p99 step time (s)")):
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        w = 0.16
        for i, m in enumerate(methods):
            xs, ys, es = [], [], []
            for j, sc in enumerate(scen):
                s = table[sc].get(m, {}).get(metric)
                if s and s["n"]:
                    xs.append(j + (i - 2) * w)
                    ys.append(s["mean"])
                    es.append((s["ci95_hi"] - s["ci95_lo"]) / 2 if s["n"] > 1 else 0)
            ax.bar(
                xs,
                ys,
                width=w * 0.9,
                color=plotstyle.METHOD_COLOR[m],
                label=m,
                yerr=es,
                error_kw={"elinewidth": 0.8, "ecolor": plotstyle.TEXT2, "capsize": 0},
            )
        ax.set_xticks(range(len(scen)), [s.replace("_", "\n") for s in scen])
        ax.set_ylabel(label)
        ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.15))
        fig.tight_layout()
        fig.savefig(figs / f"{metric}_by_scenario.png", dpi=180)
        plt.close(fig)
    # step-time traces, seed 0, per scenario
    for sc in scen:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        for m in methods:
            rs = [r for r in runs[sc].get(m, []) if r["seed"] == 0]
            if not rs or not rs[0]["step_walls"]:
                continue
            w0 = rs[0]["step_walls"][0] - rs[0]["step_times"][0]
            ax.plot(
                np.array(rs[0]["step_walls"]) - w0,
                rs[0]["step_times"],
                color=plotstyle.METHOD_COLOR[m],
                label=m,
                linewidth=1.2,
            )
        ax.set_xlabel("wall time since first step (s)")
        ax.set_ylabel("step time (s)")
        ax.set_title(f"{sc} (seed 0)", loc="left")
        ax.legend(ncol=5, fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / f"trace_{sc}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
