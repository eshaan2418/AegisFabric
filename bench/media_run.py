"""M6 campaign: joint vs independent media/inference adaptation under userspace-emulated impairments.

  uv run python -m bench.media_run --trials 5            # full M6 campaign (~30 min)
  uv run python -m bench.media_run --trials 2 --quick    # reproduce-quick subset

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


def aggregate() -> dict:
    rows = []
    for p in sorted((RESULTS / "m6" / "trials").glob("*.json")):
        rows.append(json.loads(p.read_text())["data"])
    agg: dict = {"conditions": {}, "tests": {}}
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
        agg["conditions"][name]["stt"] = sorted({r["stt"] for r in rs})
    pvals = {}
    for grp in ("clean", "lossy", "bwdrop", "overload"):
        a = [r for r in rows if r["condition"] == f"{grp}-joint"]
        b = [r for r in rows if r["condition"] == f"{grp}-independent"]
        if len(a) >= 2 and len(b) >= 2:
            for m in ("vmaf_mean", "glass_to_answer_mean_s", "stall_time_s", "answer_correct_frac"):
                xa = [r[m] for r in a if r.get(m) is not None]
                xb = [r[m] for r in b if r.get(m) is not None]
                if len(xa) >= 2 and len(xb) >= 2 and (np.std(xa) > 0 or np.std(xb) > 0):
                    t = welch(xa, xb)
                    agg["tests"][f"{grp}:{m}"] = t
                    pvals[f"{grp}:{m}"] = t["p_welch"]
    for k, v in holm(pvals).items():
        agg["tests"][k]["p_holm"] = v
    write_result("m6/summary.json", agg)
    return agg


def figures(agg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = RESULTS / "m6" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    names = list(agg["conditions"])
    for m, label in [
        ("glass_to_answer_mean_s", "glass-to-answer (s)"),
        ("vmaf_mean", "VMAF (rendered)"),
        ("stall_time_s", "stall time (s)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        means = [agg["conditions"][n][m]["mean"] for n in names]
        err = [(agg["conditions"][n][m]["ci95_hi"] - agg["conditions"][n][m]["ci95_lo"]) / 2 for n in names]
        colors = [
            "#4878a8" if "independent" in n else "#d0743c" if "joint" in n else "#8c8c8c" for n in names
        ]
        ax.bar(range(len(names)), means, yerr=err, color=colors, capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"M6: {label}, mean ± 95% CI")
        fig.tight_layout()
        fig.savefig(figs / f"{m}.png", dpi=140)
        plt.close(fig)


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
