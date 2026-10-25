"""Claims for M1 (baselines), M2 (convergence), M3 (controller), M5 rotation and the M10 demo.

Ids are letters only (LaTeX macro names). See bench/claims.py.
"""

from __future__ import annotations

from bench.claims import claim

NETS = {"loopback": "Lo", "wan": "Wan"}
MODES = {"ddp": "Ddp", "fsdp": "Fsdp", "hsdp": "Hsdp"}
M1_METRICS = {
    "reached_target_s": ("Ttt", "{:.0f}"),
    "step_mean_s": ("Step", "{:.2f}"),
    "coll_share": ("Coll", "{:.0f}"),
    "tok_s_mean": ("Tok", "{:.0f}"),
    "mfu_mean": ("Mfu", "{:.0f}"),
}


def _m1(net, mode, metric, stat):
    pct = metric in ("coll_share", "mfu_mean")

    def get(d):
        v = d["summary"][f"{net}/{mode}"][metric][stat]
        return v * 100 if pct else v

    return get


for _net, _N in NETS.items():
    for _mode, _M in MODES.items():
        for _metric, (_K, _fmt) in M1_METRICS.items():
            claim(f"mOne{_N}{_M}{_K}", "m1/baselines.json", _fmt, f"M1 {_net}/{_mode} {_metric} mean")(
                _m1(_net, _mode, _metric, "mean")
            )
            # half-width of the 95% CI
            claim(f"mOne{_N}{_M}{_K}Ci", "m1/baselines.json", _fmt, f"M1 {_net}/{_mode} {_metric} CI±")(
                (lambda g_lo, g_hi: lambda d: (g_hi(d) - g_lo(d)) / 2)(
                    _m1(_net, _mode, _metric, "ci95_lo"), _m1(_net, _mode, _metric, "ci95_hi")
                )
            )
        claim(f"mOne{_N}{_M}N", "m1/baselines.json", "{:d}", "M1 runs reaching target")(
            (lambda n, m: lambda d: d["summary"][f"{n}/{m}"]["n_reached"])(_net, _mode)
        )


@claim("mOneSeeds", "m1/baselines.json", "{:d}")
def _m1_seeds(d):
    return len({r["seed"] for r in d["runs"]})


@claim("mOneWanOverLoDdp", "m1/baselines.json", "{:.1f}", "WAN/loopback DDP step-time ratio")
def _m1_ratio(d):
    s = d["summary"]
    return s["wan/ddp"]["step_mean_s"]["mean"] / s["loopback/ddp"]["step_mean_s"]["mean"]


# --- M2: convergence per failure type ------------------------------------------------------------------
KINDS = {
    "link_down": "LinkDown",
    "blackhole": "Blackhole",
    "flap": "Flap",
    "bgp_reset": "BgpReset",
    "bw_collapse": "BwCollapse",
    "latency_spike": "LatSpike",
}
M2_KEYS = {
    "outage_s": ("Outage", "{:.2f}"),
    "fib_change_s": ("Fib", "{:.2f}"),
    "restore_fib_s": ("RestoreFib", "{:.2f}"),
    "restore_outage_s": ("RestoreOutage", "{:.2f}"),
    "bgp_down_detect_s": ("BgpDetect", "{:.2f}"),
    "bgp_reestablish_s": ("BgpReest", "{:.1f}"),
    "rtt_before_ms": ("RttBefore", "{:.1f}"),
    "rtt_during_ms": ("RttDuring", "{:.0f}"),
    "tput_before_mbit": ("TputBefore", "{:.0f}"),
    "tput_during_mbit": ("TputDuring", "{:.1f}"),
}
M2_APPLIES = {
    "link_down": ["outage_s", "fib_change_s", "restore_fib_s", "restore_outage_s"],
    "blackhole": ["outage_s", "fib_change_s", "bgp_down_detect_s"],
    "flap": ["outage_s", "fib_change_s", "restore_fib_s", "restore_outage_s"],
    "bgp_reset": ["outage_s", "fib_change_s", "bgp_reestablish_s"],
    "bw_collapse": ["rtt_before_ms", "rtt_during_ms", "tput_before_mbit", "tput_during_mbit"],
    "latency_spike": ["rtt_before_ms", "rtt_during_ms", "tput_before_mbit", "tput_during_mbit"],
}
for _kind, _K in KINDS.items():
    for _key in M2_APPLIES[_kind]:
        _n, _fmt = M2_KEYS[_key]
        claim(f"mTwo{_K}{_n}", "m2/convergence.json", _fmt, f"M2 {_kind} {_key} mean")(
            (lambda k, q: lambda d: d["summary"][k][q]["mean"])(_kind, _key)
        )
        claim(f"mTwo{_K}{_n}Ci", "m2/convergence.json", _fmt, f"M2 {_kind} {_key} CI±")(
            (lambda k, q: lambda d: (d["summary"][k][q]["ci95_hi"] - d["summary"][k][q]["ci95_lo"]) / 2)(
                _kind, _key
            )
        )


@claim("mTwoTrials", "m2/convergence.json", "{:d}")
def _m2_trials(d):
    return d["summary"]["link_down"]["outage_s"]["n"]


@claim("mTwoInitConv", "m2/convergence.json", "{:.1f}", "initial BGP convergence of the lab (s)")
def _m2_init(d):
    return d["initial_convergence_s"]
