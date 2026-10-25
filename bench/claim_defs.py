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
    "blackhole": ["outage_s", "fib_change_s", "restore_fib_s", "restore_outage_s"],
    "flap": ["outage_s", "fib_change_s"],
    "bgp_reset": ["outage_s", "fib_change_s", "bgp_down_detect_s", "bgp_reestablish_s"],
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


# --- M3 calibration and ring-structure validation -------------------------------------------------------
@claim("ringContig", "m3/ring_micro.json", "{:.2f}", "all-reduce s, contiguous ring order")
def _ring_c(d):
    return d["summary"]["contiguous"]["allreduce_s"]["mean"]


@claim("ringInter", "m3/ring_micro.json", "{:.2f}", "all-reduce s, interleaved ring order")
def _ring_i(d):
    return d["summary"]["interleaved"]["allreduce_s"]["mean"]


@claim("ringRatio", "m3/ring_micro.json", "{:.2f}")
def _ring_r(d):
    return d["summary"]["ratio_interleaved_over_contiguous"]["mean"]


@claim("ringRatioCi", "m3/ring_micro.json", "{:.2f}")
def _ring_rci(d):
    s = d["summary"]["ratio_interleaved_over_contiguous"]
    return (s["ci95_hi"] - s["ci95_lo"]) / 2


@claim("ringPredRatio", "m3/ring_micro.json", "{:.2f}")
def _ring_pr(d):
    return d["summary"]["pred_ratio"]


@claim("ringPredContig", "m3/ring_micro.json", "{:.2f}")
def _ring_pc(d):
    return d["summary"]["contiguous"]["pred_s"]


@claim("ringPredInter", "m3/ring_micro.json", "{:.2f}")
def _ring_pi(d):
    return d["summary"]["interleaved"]["pred_s"]


@claim("ringTrials", "m3/ring_micro.json", "{:d}")
def _ring_n(d):
    return d["summary"]["contiguous"]["allreduce_s"]["n"]


for _k, _fmt in (("compute_s", "{:.2f}"), ("restart_s", "{:.1f}")):
    claim("cal" + "".join(w.title() for w in _k.split("_")), "m3/calibration.json", _fmt)(
        (lambda k: lambda d: d[k])(_k)
    )
claim("calGradMB", "m3/calibration.json", "{:.1f}")(lambda d: d["grad_bytes"] / 1e6)
claim("calCkptMB", "m3/calibration.json", "{:.1f}")(lambda d: d["ckpt_bytes"] / 1e6)
claim("calKappaA", "m3/calibration.json", "{:.2f}")(lambda d: d["kappa_plane"]["a"])
claim("calKappaB", "m3/calibration.json", "{:.2f}")(lambda d: d["kappa_plane"]["b"])


# --- M3 main campaign (results/m3/summary.json from bench.m3_analyze) ----------------------------------
SCEN = {
    "steady": "Steady",
    "link_down": "LinkDown",
    "blackhole": "Blackhole",
    "flap": "Flap",
    "bw_collapse": "BwCollapse",
    "latency_spike": "LatSpike",
    "bgp_reset": "BgpReset",
    "site_degradation": "SiteDeg",
}
METH = {"static": "Static", "ecmp": "Ecmp", "independent": "Indep", "joint": "Joint", "oracle": "Oracle"}
M3_ROW = {
    "total_s": ("Total", "{:.0f}"),
    "step_p99_s": ("Pnn", "{:.2f}"),
    "step_std_s": ("Std", "{:.2f}"),
    "ttq_s": ("Ttq", "{:.0f}"),
}


def _row(sc, m, k, stat="mean"):
    def get(d):
        s = d["table"][sc][m][k]
        if stat == "ci":
            return (s["ci95_hi"] - s["ci95_lo"]) / 2
        return s[stat]

    return get


for _sc, _S in SCEN.items():
    for _m, _Mm in METH.items():
        for _k, (_K, _fmt) in M3_ROW.items():
            claim(f"mThree{_S}{_Mm}{_K}", "m3/summary.json", _fmt)(_row(_sc, _m, _k))
            claim(f"mThree{_S}{_Mm}{_K}Ci", "m3/summary.json", _fmt)(_row(_sc, _m, _k, "ci"))
        claim(f"mThree{_S}{_Mm}Mig", "m3/summary.json", "{:.1f}")(
            (lambda s, m: lambda d: d["table"][s][m]["migrations_mean"])(_sc, _m)
        )
        claim(f"mThree{_S}{_Mm}N", "m3/summary.json", "{:d}")(
            (lambda s, m: lambda d: d["table"][s][m]["n"])(_sc, _m)
        )
        if _sc != "steady":
            claim(f"mThree{_S}{_Mm}Excess", "m3/summary.json", "{:.1f}", "time lost in 60 s after inject")(
                (lambda s, m: lambda d: d["table"][s][m]["excess_s_60"]["mean"])(_sc, _m)
            )
            claim(f"mThree{_S}{_Mm}Rec", "m3/summary.json", "{:.1f}")(
                (lambda s, m: lambda d: d["table"][s][m]["recovery_s"]["mean"])(_sc, _m)
            )
            claim(f"mThree{_S}{_Mm}Cens", "m3/summary.json", "{:d}", "unrecovered failure events")(
                (lambda s, m: lambda d: d["table"][s][m]["recovery_censored"])(_sc, _m)
            )
        if _m != "joint":
            for _k, _K in (("total_s", "Total"), ("step_p99_s", "Pnn"), ("excess_mean_s", "Excess")):
                if _k == "excess_mean_s" and _sc == "steady":
                    continue
                claim(f"mThree{_S}{_Mm}{_K}Impr", "m3/summary.json", "{:.0f}", "joint rel. improvement %")(
                    (
                        lambda s, m, k: (
                            lambda d: 100 * d["tests"]["baselines"][k][f"{s}|{m}"]["rel_improvement_paired"]
                        )
                    )(_sc, _m, _k)
                )
                claim(f"mThree{_S}{_Mm}{_K}Holm", "m3/summary.json", "{:.2g}", "Holm-adjusted p")(
                    (lambda s, m, k: lambda d: d["tests"]["baselines"][k][f"{s}|{m}"]["p_holm"])(_sc, _m, _k)
                )


def _get(m, k, f):
    return lambda d: d["pooled"][m][k][f]


for _m, _Mm in METH.items():
    if _m == "joint":
        continue
    for _k, _K in (
        ("total_s", "Total"),
        ("step_p99_s", "Pnn"),
        ("step_std_s", "Std"),
        ("excess_mean_s", "Excess"),
        ("ttq_s", "Ttq"),
    ):
        claim(f"pool{_Mm}{_K}", "m3/summary.json", "{:.0f}", "pooled joint rel. improvement %")(
            (lambda g: lambda d: 100 * g(d))(_get(_m, _k, "mean_rel_improvement"))
        )
        claim(f"pool{_Mm}{_K}Lo", "m3/summary.json", "{:.0f}")(
            (lambda g: lambda d: 100 * g(d))(_get(_m, _k, "ci95_lo"))
        )
        claim(f"pool{_Mm}{_K}Hi", "m3/summary.json", "{:.0f}")(
            (lambda g: lambda d: 100 * g(d))(_get(_m, _k, "ci95_hi"))
        )
        claim(f"pool{_Mm}{_K}P", "m3/summary.json", "{:.2g}")(_get(_m, _k, "p_wilcoxon"))
        claim(f"pool{_Mm}{_K}Frac", "m3/summary.json", "{:.0f}")(
            (lambda g: lambda d: 100 * g(d))(_get(_m, _k, "frac_joint_better"))
        )
        claim(f"pool{_Mm}{_K}N", "m3/summary.json", "{:d}")(_get(_m, _k, "n_pairs"))


@claim("mThreeRuns", "m3/summary.json", "{:d}")
def _m3_runs(d):
    return sum(r["n"] for sc in d["table"].values() for m, r in sc.items() if not m.startswith("joint-"))


@claim("mThreeCrashes", "m3/summary.json", "{:d}")
def _m3_crashes(d):
    return sum(r["crashes"] for sc in d["table"].values() for r in sc.values())


@claim("mThreeTimedOut", "m3/summary.json", "{:d}")
def _m3_to(d):
    return sum(r["timed_out"] for sc in d["table"].values() for r in sc.values())


# --- M3 ablations (joint vs joint-without-X; 3 seeds; family "ablations") ------------------------------
ABL = {
    "joint-no-util": "NoUtil",
    "joint-no-speed": "NoSpeed",
    "joint-no-rtt": "NoRtt",
    "joint-no-coll": "NoColl",
    "joint-no-ckpt": "NoCkpt",
    "joint-nomig": "NoMig",
}
ABL_SCEN = {
    "steady": "Steady",
    "bw_collapse": "BwCollapse",
    "latency_spike": "LatSpike",
    "site_degradation": "SiteDeg",
}
for _v, _V in ABL.items():
    for _sc, _S in ABL_SCEN.items():
        claim(f"abl{_V}{_S}Total", "m3/summary.json", "{:.0f}")(_row(_sc, _v, "total_s"))
        claim(
            f"abl{_V}{_S}Impr", "m3/summary.json", "{:.0f}", "% by which full joint beats the ablated variant"
        )(
            (
                lambda s, v: (
                    lambda d: 100 * d["tests"]["ablations"]["total_s"][f"{s}|{v}"]["rel_improvement_paired"]
                )
            )(_sc, _v)
        )
        claim(f"abl{_V}{_S}Holm", "m3/summary.json", "{:.2g}")(
            (lambda s, v: lambda d: d["tests"]["ablations"]["total_s"][f"{s}|{v}"]["p_holm"])(_sc, _v)
        )
    claim(f"abl{_V}Pool", "m3/summary.json", "{:.0f}")(
        (lambda v: lambda d: 100 * d["pooled"][v]["total_s"]["mean_rel_improvement"])(_v)
    )
    claim(f"abl{_V}PoolLo", "m3/summary.json", "{:.0f}")(
        (lambda v: lambda d: 100 * d["pooled"][v]["total_s"]["ci95_lo"])(_v)
    )
    claim(f"abl{_V}PoolHi", "m3/summary.json", "{:.0f}")(
        (lambda v: lambda d: 100 * d["pooled"][v]["total_s"]["ci95_hi"])(_v)
    )


# --- M5: negative tests and rotation under load ---------------------------------------------------------
claim("negCases", "m5/negative.json", "{:d}")(lambda d: d["n_cases"])
claim("negRejected", "m5/negative.json", "{:d}")(lambda d: d["n_rejected"])
ROT = {"none": "None", "plaintext": "Plain", "mtls": "Mtls", "mtls-rot": "Rot"}
for _c, _C in ROT.items():
    claim(f"rot{_C}Step", "m5/rotation.json", "{:.2f}")(
        (lambda c: lambda d: d["summary"][c]["step_mean_s"]["mean"])(_c)
    )
    claim(f"rot{_C}StepCi", "m5/rotation.json", "{:.2f}")(
        (
            lambda c: (
                lambda d: (
                    (d["summary"][c]["step_mean_s"]["ci95_hi"] - d["summary"][c]["step_mean_s"]["ci95_lo"])
                    / 2
                )
            )
        )(_c)
    )
    claim(f"rot{_C}Failed", "m5/rotation.json", "{:d}")(
        (lambda c: lambda d: d["summary"][c]["failed_ranks_total"])(_c)
    )
    claim(f"rot{_C}RpcErr", "m5/rotation.json", "{:d}")(
        (lambda c: lambda d: d["summary"][c]["rpc_errors_total"])(_c)
    )
    claim(f"rot{_C}Rpcs", "m5/rotation.json", "{:d}")((lambda c: lambda d: d["summary"][c]["rpcs_total"])(_c))
    if _c != "none":
        claim(f"rot{_C}VsNoneP", "m5/rotation.json", "{:.2g}")(
            (lambda c: lambda d: d["summary"][c]["vs_none"]["p_welch"])(_c)
        )
        claim(f"rot{_C}VsNone", "m5/rotation.json", "{:.1f}")(
            (lambda c: lambda d: 100 * d["summary"][c]["vs_none"]["rel_diff"])(_c)
        )
    if _c.startswith("mtls"):
        claim(f"rot{_C}Hs", "m5/rotation.json", "{:.1f}")(
            (lambda c: lambda d: d["summary"][c]["handshake_ms"]["mean"])(_c)
        )
        claim(f"rot{_C}HsPnn", "m5/rotation.json", "{:.1f}")(
            (lambda c: lambda d: d["summary"][c]["handshake_ms"]["p99"])(_c)
        )
        claim(f"rot{_C}RpcLat", "m5/rotation.json", "{:.1f}")(
            (lambda c: lambda d: d["summary"][c]["rpc_lat_ms_mean"]["mean"])(_c)
        )
claim("rotRenewals", "m5/rotation.json", "{:d}")(lambda d: d["summary"]["mtls-rot"]["renewals_total"])
claim("rotRenewMs", "m5/rotation.json", "{:.1f}")(lambda d: d["summary"]["mtls-rot"]["renewal_ms"]["mean"])
claim("rotTrials", "m5/rotation.json", "{:d}")(lambda d: d["summary"]["none"]["step_mean_s"]["n"])
