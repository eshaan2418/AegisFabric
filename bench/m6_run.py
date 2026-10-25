"""M6 campaign: joint vs independent media/inference adaptation under userspace-emulated impairments.

  uv run python -m bench.m6_run --trials 5            # full M6 campaign (~30 min)
  uv run python -m bench.m6_run --trials 2 --quick    # reproduce-quick subset

Each trial writes results/m6/trials/<condition>_s<seed>.json; aggregates go to results/m6/summary.json and
figures to results/m6/figs/. Network impairment: seeded in-process link emulation (media/netem.py), NOT kernel netem; executed in
virtual time (media/vloop.py) with real codec/STT/vision compute charged at its measured CPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SVT_LOG", "1")

import numpy as np  # noqa: E402

from bench.meta import RESULTS, read_result, write_result  # noqa: E402
from bench.stats import holm, summarize, welch  # noqa: E402
from media.inference import StubVisionModel, WhisperStt  # noqa: E402
from media.netem import PROFILES  # noqa: E402
from media.session import TrialConfig, run_trial, summarize_trial  # noqa: E402
from media.source import build_source  # noqa: E402
from media.vmaf import rendered_vmaf  # noqa: E402

CONDITIONS = [
    # (name, profile, policy, vision_ms)
    ("clean-independent", "clean", "independent", 50.0),
    ("clean-joint", "clean", "joint", 50.0),
    ("lossy-independent", "lossy", "independent", 50.0),
    ("lossy-joint", "lossy", "joint", 50.0),
    ("bwdrop-independent", "bwdrop", "independent", 50.0),
    ("bwdrop-joint", "bwdrop", "joint", 50.0),
    ("jitter-independent", "jitter", "independent", 50.0),
    ("jitter-joint", "jitter", "joint", 50.0),
    ("overload-independent", "clean", "independent", 150.0),
    ("overload-joint", "clean", "joint", 150.0),
    ("overload-blocking", "clean", "blocking", 150.0),
]
QUICK = ["clean-independent", "clean-joint", "overload-independent", "overload-joint"]
METRICS = [
    "vmaf_mean",
    "sent_kbps",
    "glass_to_answer_mean_s",
    "answer_correct_frac",
    "stall_time_s",
    "stalls",
    "dropped_network_frac",
    "dropped_backpressure",
    "glass_to_inference_p95_s",
    "av_skew_ms_median_abs",
]


def run(trials: int, names: list[str], duration: float) -> None:
    qt = read_result("m6/calibration.json")["data"]["table"]
    # virtual-time mode: models run in-process, single-threaded; their CPU time is their latency
    stt = WhisperStt(threads=1)
    visions = {}
    for name, prof, pol, vms in [c for c in CONDITIONS if c[0] in names]:
        for seed in range(trials):
            out = RESULTS / "m6" / "trials" / f"{name}_s{seed}.json"
            if out.exists():
                continue
            src = build_source(seed, duration_s=duration)
            cfg = TrialConfig(PROFILES[prof], pol, seed=seed, vision_ms=vms)
            t = time.time()
            load0 = os.getloadavg()
            vision = visions.setdefault(vms, StubVisionModel(vms, seed=0, threads=1, pad=False))
            log = run_trial(src, cfg, vision, stt, qt, mode="virtual")
            load1 = os.getloadavg()
            v = rendered_vmaf(src.frames, log.display, log.decoded, 1.0, duration)
            s = summarize_trial(src, cfg, log, v)
            s.update(
                condition=name,
                stt=stt.name,
                speech=src.speech,
                wall_s=time.time() - t,
                loadavg_1m=[load0[0], load1[0]],
                host_cpus=os.cpu_count(),
                netem="in-process emulated link (media/netem.py), virtual time",
                mode="virtual-time; compute latency = measured single-thread CPU time",
                machine_note="dev run, machine shared",
            )
            write_result(
                out, s, config={"condition": name, "duration_s": duration, "vision_ms": vms}, seed=seed
            )
            print(
                name,
                seed,
                {k: s[k] for k in ("vmaf_mean", "glass_to_answer_mean_s", "stall_time_s", "sent_kbps")},
            )


GROUPS = ["clean", "lossy", "bwdrop", "jitter", "overload"]
NET_GROUPS = ["lossy", "bwdrop", "jitter"]  # the impairment profiles (clean = control, overload = compute)
TEST_METRICS = [
    "glass_to_answer_mean_s",
    "vmaf_mean",
    "stall_time_s",
    "answer_correct_frac",
    "sent_kbps",
    "dropped_network_frac",
    "dropped_backpressure",
    "glass_to_inference_p95_s",
]


def _test(a: list[dict], b: list[dict], m: str) -> dict | None:
    """a vs b: Welch + Mann-Whitney (bench.stats.welch) and a paired t-test over seeds present in both
    (same seed = same content and same link-impairment RNG stream)."""
    from scipy import stats as st

    xa = [r[m] for r in a if r.get(m) is not None]
    xb = [r[m] for r in b if r.get(m) is not None]
    if len(xa) < 2 or len(xb) < 2 or (np.std(xa) == 0 and np.std(xb) == 0):
        return None
    t = welch(xa, xb)
    ia = {(r["profile"], r["seed"]): r[m] for r in a if r.get(m) is not None}
    ib = {(r["profile"], r["seed"]): r[m] for r in b if r.get(m) is not None}
    keys = sorted(set(ia) & set(ib))
    d = [ia[k] - ib[k] for k in keys]
    t["n_a"], t["n_b"], t["n_paired"] = len(xa), len(xb), len(keys)
    t["mean_a"], t["mean_b"] = float(np.mean(xa)), float(np.mean(xb))
    t["mean_diff_paired"] = float(np.mean(d)) if d else None
    t["p_paired"] = (
        float(st.ttest_rel([ia[k] for k in keys], [ib[k] for k in keys]).pvalue)
        if len(d) >= 2 and np.std(d) > 0
        else None
    )
    return t


def aggregate() -> dict:
    rows = []
    for p in sorted((RESULTS / "m6" / "trials").glob("*.json")):
        rows.append(json.loads(p.read_text())["data"])
    agg: dict = {
        "conditions": {},
        "tests": {},
        "pooled_tests": {},
        "ablation_tests": {},
        "note": "tests: a = joint, b = independent (ablation: a = joint, b = blocking)",
    }
    loads = [x for r in rows for x in r.get("loadavg_1m", [])]
    agg["loadavg_1m"] = (
        {"min": min(loads), "max": max(loads), "mean": float(np.mean(loads))} if loads else None
    )
    for name, *_ in CONDITIONS:
        rs = [r for r in rows if r["condition"] == name]
        if not rs:
            continue
        agg["conditions"][name] = {
            m: summarize([r[m] for r in rs if r.get(m) is not None]).as_dict() for m in METRICS
        }
        agg["conditions"][name]["n"] = len(rs)
        agg["conditions"][name]["seeds"] = sorted(r["seed"] for r in rs)
        agg["conditions"][name]["stt"] = sorted({r["stt"] for r in rs})
    pvals = {}
    for grp in GROUPS:
        a = [r for r in rows if r["condition"] == f"{grp}-joint"]
        b = [r for r in rows if r["condition"] == f"{grp}-independent"]
        for m in TEST_METRICS:
            t = _test(a, b, m)
            if t is not None:
                agg["tests"][f"{grp}:{m}"] = t
                pvals[f"{grp}:{m}"] = t["p_welch"]
    for k, v in holm(pvals).items():
        agg["tests"][k]["p_holm"] = v
    # pooled over the three impairment profiles, paired by (profile, seed)
    a = [r for r in rows if r["condition"] in {f"{g}-joint" for g in NET_GROUPS}]
    b = [r for r in rows if r["condition"] in {f"{g}-independent" for g in NET_GROUPS}]
    pp = {}
    for m in TEST_METRICS:
        t = _test(a, b, m)
        if t is not None:
            agg["pooled_tests"][m] = t
            if t["p_paired"] is not None:
                pp[m] = t["p_paired"]
    for k, v in holm(pp).items():
        agg["pooled_tests"][k]["p_paired_holm"] = v
    # backpressure ablation: blocking queue vs joint (latest-wins) under inference overload
    a = [r for r in rows if r["condition"] == "overload-joint"]
    b = [r for r in rows if r["condition"] == "overload-blocking"]
    for m in ("stall_time_s", "glass_to_answer_mean_s", "vmaf_mean", "blocked_s"):
        t = _test(a, b, m)
        if t is not None:
            agg["ablation_tests"][f"joint_vs_blocking:{m}"] = t
    write_result(
        "m6/summary.json",
        agg,
        config={"trials_dir": "results/m6/trials", "tests": "Welch+MWU, paired t over seeds, Holm"},
    )
    return agg


LABELS = {
    "glass_to_answer_mean_s": "glass-to-answer (s)",
    "vmaf_mean": "VMAF, rendered stream",
    "stall_time_s": "stall time (s per 24 s)",
    "sent_kbps": "video sent (kbps)",
    "dropped_network_frac": "frames lost/late (fraction)",
    "glass_to_inference_p95_s": "glass-to-inference p95 (s)",
}


def figures(agg: dict) -> None:
    import matplotlib.pyplot as plt

    from bench import plotstyle

    plotstyle.apply()
    figs = RESULTS / "m6" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    colors = {
        "independent": plotstyle.METHOD_COLOR["independent"],
        "joint": plotstyle.METHOD_COLOR["joint"],
        "blocking": plotstyle.SLOTS[6],
    }
    pols = ["independent", "joint", "blocking"]
    groups = [g for g in GROUPS if any(f"{g}-{p}" in agg["conditions"] for p in pols)]
    table = ["# M6 figure tables (mean ± 95% CI half-width, n seeds)", ""]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (m, label) in zip(axes.ravel(), LABELS.items(), strict=True):
        table += [f"## {label}", "", "| profile | " + " | ".join(pols) + " |", "|---|---|---|---|"]
        w = 0.26
        for k, pol in enumerate(pols):
            xs, ys, es = [], [], []
            for i, g in enumerate(groups):
                c = agg["conditions"].get(f"{g}-{pol}")
                if not c or c[m]["n"] == 0:
                    continue
                xs.append(i + (k - 1) * w)
                ys.append(c[m]["mean"])
                es.append((c[m]["ci95_hi"] - c[m]["ci95_lo"]) / 2 if c[m]["n"] > 1 else 0)
            if xs:
                ax.bar(
                    xs,
                    ys,
                    width=w - 0.03,
                    yerr=es,
                    color=colors[pol],
                    label=pol,
                    capsize=2,
                    error_kw={"elinewidth": 1, "ecolor": plotstyle.TEXT2},
                )
        for g in groups:
            cells = []
            for pol in pols:
                c = agg["conditions"].get(f"{g}-{pol}")
                if c and c[m]["n"]:
                    h = (c[m]["ci95_hi"] - c[m]["ci95_lo"]) / 2 if c[m]["n"] > 1 else float("nan")
                    cells.append(f"{c[m]['mean']:.3g} ± {h:.2g} (n={c[m]['n']})")
                else:
                    cells.append("-")
            table.append(f"| {g} | " + " | ".join(cells) + " |")
        table.append("")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_title(label, fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("M6: joint vs independent adaptation, mean ± 95% CI over seeds (virtual time)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "m6_metrics.png", dpi=150)
    plt.close(fig)
    (figs / "TABLES.md").write_text("\n".join(table) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    names = QUICK if a.quick else [c[0] for c in CONDITIONS]
    if not a.aggregate_only:
        run(a.trials, names, a.duration)
    agg = aggregate()
    figures(agg)
    for n, c in agg["conditions"].items():
        print(n, {m: round(c[m]["mean"], 3) for m in ("vmaf_mean", "glass_to_answer_mean_s", "stall_time_s")})


if __name__ == "__main__":
    main()
