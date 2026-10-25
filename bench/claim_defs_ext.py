"""Claims for the supporting subsystems and extensions (paper section `sec:ext`).

M4 checkpoint engine, M5 envelope overhead, M6 media, M7 kernels, M8 post-training/privacy/federated and
M9 TLC runs. Every claim reads one result file under results/; the paper uses the generated `\\C<id>` macros.
Ids are letters only: seeds are spelled A..E (seed 0..4), epsilon targets as words.
"""

from __future__ import annotations

from bench.claims import claim


def _sci(v: float, digits: int = 1) -> str:
    """LaTeX scientific notation for use inside math mode, e.g. 4.0\\times10^{12}."""
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def _mean(x) -> float:
    return x["mean"]


def _half(x) -> float:
    return (x["ci95_hi"] - x["ci95_lo"]) / 2


def _reg(id_: str, source: str, fn, fmt: str = "{:.2f}", desc: str = "") -> None:
    claim(id_, source, fmt, desc)(fn)


SEEDS = "ABCDE"  # seed 0..4

# ---------------------------------------------------------------- M4: crash campaigns
for _camp, _file in (("Kill", "crash_kill9"), ("Torn", "crash_torn"), ("Flip", "crash_bitflip")):
    _src = f"m4/{_file}.json"
    _reg(f"ckpt{_camp}Trials", _src, lambda d: d["trials"], "{:d}", f"{_file}: trials")
    for _o in ("ok", "detected", "undetected"):
        _reg(
            f"ckpt{_camp}{_o.capitalize()}",
            _src,
            lambda d, o=_o: d["outcomes"][o],
            "{:d}",
            f"{_file}: outcome {_o}",
        )
for _camp, _file in (("Torn", "crash_torn"), ("Flip", "crash_bitflip")):
    for _t in ("index", "wal", "segment"):
        for _o in ("ok", "detected"):
            _reg(
                f"ckpt{_camp}{_t.capitalize()}{_o.capitalize()}",
                f"m4/{_file}.json",
                lambda d, t=_t, o=_o: d["by_target"][t][o],
                "{:d}",
                f"{_file}: target {_t}, outcome {_o}",
            )

_K9 = "m4/crash_kill9.json"
_reg("ckptKillSaves", _K9, lambda d: d["total_saves"], "{:d}", "saves performed in kill -9 trials")
_reg("ckptKillGcs", _K9, lambda d: d["total_gcs"], "{:d}", "GC cycles in kill -9 trials")
_reg("ckptKillLatestAcked", _K9, lambda d: d["durability"]["latest_eq_acked"], "{:d}", "latest == last acked")
_reg(
    "ckptKillLatestUnacked",
    _K9,
    lambda d: d["durability"]["latest_eq_attempted_unacked"],
    "{:d}",
    "latest == durable but unacked",
)
_reg(
    "ckptKillNoSave",
    _K9,
    lambda d: (
        d["trials"] - d["durability"]["latest_eq_acked"] - d["durability"]["latest_eq_attempted_unacked"]
    ),
    "{:d}",
    "kill -9 trials in which no save had completed",
)
_reg(
    "ckptKillUring",
    _K9,
    lambda d: sum(t["io"] == "uring" for t in d["trials_detail"]),
    "{:d}",
    "kill -9 trials using io_uring",
)
_reg(
    "ckptKillPwrite",
    _K9,
    lambda d: sum(t["io"] == "pwrite" for t in d["trials_detail"]),
    "{:d}",
    "kill -9 trials using pwrite",
)
_reg(
    "ckptKillRecoveryMs",
    _K9,
    lambda d: d["recovery_ms_mean"],
    "{:.0f}",
    "mean server recovery after kill -9 (ms)",
)

# ---------------------------------------------------------------- M4: benchmark matrix (bench.json)
_B = "m4/bench.json"
_METHODS = {
    "Torch": "torch_save",
    "Dcp": "dcp",
    "Uring": "aegis_uring",
    "Pwrite": "aegis_pwrite",
    "Sync": "aegis_uring_sync",
}
_METRICS = {
    "Stall": ("stall_s", "{:.3f}"),
    "Durable": ("durable_s", "{:.2f}"),
    "Recovery": ("recovery_s", "{:.2f}"),
    "Wamp": ("write_amp_device", "{:.2f}"),
    "Dedup": ("dedup_ratio", "{:.2f}"),
}
for _scn in ("full", "frozen"):
    for _mn, _mk in _METHODS.items():
        for _metn, (_metk, _fmt) in _METRICS.items():
            _key = f"xfs/{_scn}/{_mk}"
            _base = f"ckpt{_metn}Xfs{_scn.capitalize()}{_mn}"
            _reg(
                _base, _B, lambda d, k=_key, m=_metk: _mean(d["summary"][k][m]), _fmt, f"{_key} {_metk} mean"
            )
            _reg(
                _base + "Ci",
                _B,
                lambda d, k=_key, m=_metk: _half(d["summary"][k][m]),
                _fmt,
                f"{_key} {_metk} 95% CI half-width",
            )

_reg(
    "ckptBenchN", _B, lambda d: d["summary"]["xfs/full/torch_save"]["stall_s"]["n"], "{:d}", "trials per cell"
)
_reg(
    "ckptStateFullMB",
    _B,
    lambda d: d["summary"]["xfs/full/torch_save"]["state_bytes"] / 1e6,
    "{:.1f}",
    "state size, full (MB)",
)
_reg(
    "ckptStateFrozenMB",
    _B,
    lambda d: d["summary"]["xfs/frozen/torch_save"]["state_bytes"] / 1e6,
    "{:.1f}",
    "state size, frozen (MB)",
)


def _overlap(d, fn):
    return fn(
        v["overlap_slowdown"]["mean"]
        for k, v in d["summary"].items()
        if "overlap_slowdown" in v and not k.endswith("_sync")
    )


_reg("ckptOverlapMin", _B, lambda d: _overlap(d, min), "{:.1f}", "min step slowdown during background flush")
_reg("ckptOverlapMax", _B, lambda d: _overlap(d, max), "{:.1f}", "max step slowdown during background flush")


def _sync_over_async(d, fn):
    r = []
    for fs in ("ext4", "xfs"):
        for scn in ("full", "frozen"):
            s = d["summary"][f"{fs}/{scn}/aegis_uring_sync"]["stall_s"]["mean"]
            a = d["summary"][f"{fs}/{scn}/aegis_uring"]["stall_s"]["mean"]
            r.append(s / a)
    return fn(r)


_reg("ckptSyncRatioMin", _B, lambda d: _sync_over_async(d, min), "{:.0f}", "sync/async io_uring stall, min")
_reg("ckptSyncRatioMax", _B, lambda d: _sync_over_async(d, max), "{:.0f}", "sync/async io_uring stall, max")

# ---------------------------------------------------------------- M4: dedup (dedup.json)
_DD = "m4/dedup.json"
for _scn in ("full", "frozen", "adapter"):
    _reg(
        f"ckptDedup{_scn.capitalize()}",
        _DD,
        lambda d, s=_scn: _mean(d["summary"][f"{s}/tensor/1048576"]["dedup_ratio"]),
        "{:.2f}",
        f"dedup ratio {_scn}, per-tensor 1 MiB chunks",
    )
    _reg(
        f"ckptNewFrac{_scn.capitalize()}",
        _DD,
        lambda d, s=_scn: 100 * _mean(d["summary"][f"{s}/tensor/1048576"]["incremental_new_frac"]),
        "{:.0f}",
        f"new bytes per incremental ckpt (%), {_scn}",
    )
_reg(
    "ckptDedupFrozenBlobBig",
    _DD,
    lambda d: _mean(d["summary"]["frozen/blob/4194304"]["dedup_ratio"]),
    "{:.2f}",
    "dedup ratio frozen, torch.save byte stream, 4 MiB chunks",
)
_reg(
    "ckptDedupStateMB",
    _DD,
    lambda d: d["summary"]["full/tensor/1048576"]["state_bytes"] / 1e6,
    "{:.0f}",
    "dedup experiment state size (MB)",
)
_reg(
    "ckptDedupSeeds",
    _DD,
    lambda d: d["summary"]["full/tensor/1048576"]["dedup_ratio"]["n"],
    "{:d}",
    "dedup seeds",
)

# ---------------------------------------------------------------- M5: envelope overhead
_E = "m5/envelope_overhead.json"
for _n, _k in (("Cold", "cold_s"), ("Warm", "warm_s"), ("Verify", "cold_verify_s")):
    _reg(f"env{_n}Plain", _E, lambda d, k=_k: _mean(d["summary"]["plain"][k]), "{:.2f}", f"plaintext {_k}")
    _reg(
        f"env{_n}PlainCi", _E, lambda d, k=_k: _half(d["summary"]["plain"][k]), "{:.2f}", f"plaintext {_k} CI"
    )
    _reg(f"env{_n}Ptwo", _E, lambda d, k=_k: _mean(d["summary"]["p2"][k]), "{:.2f}", f"P2 {_k}")
    _reg(f"env{_n}PtwoCi", _E, lambda d, k=_k: _half(d["summary"]["p2"][k]), "{:.2f}", f"P2 {_k} CI")
    _reg(f"env{_n}Ratio", _E, lambda d, k=_k: d["overhead"][k]["p2_over_plain"], "{:.2f}", f"P2/plain {_k}")
    _reg(f"env{_n}Abs", _E, lambda d, k=_k: d["overhead"][k]["abs_s"], "{:.2f}", f"P2-plain {_k} (s)")
    _reg(f"env{_n}P", _E, lambda d, k=_k: d["overhead"][k]["welch"]["p_welch"], "{:.2g}", f"Welch p {_k}")
    _reg(
        f"env{_n}Pmwu",
        _E,
        lambda d, k=_k: d["overhead"][k]["welch"]["p_mannwhitney"],
        "{:.2g}",
        f"Mann-Whitney p {_k}",
    )
_reg("envTrials", _E, lambda d: d["summary"]["plain"]["cold_s"]["n"], "{:d}", "envelope trials")
_reg("envStateMB", _E, lambda d: d["state_bytes"] / 1e6, "{:.0f}", "envelope checkpoint size (MB)")
_reg(
    "envSaveEncrypt",
    _E,
    lambda d: _mean(d["summary"]["p2"]["save_encrypt_s"]),
    "{:.2f}",
    "P2 encrypt per save (s)",
)
_reg("envSaveEncryptCi", _E, lambda d: _half(d["summary"]["p2"]["save_encrypt_s"]), "{:.2f}", "CI")

# ---------------------------------------------------------------- M9: TLC
_T = "m9/summary.json"
_TLC = {
    "PoneFixed": "p1_fixed",
    "PoneBig": "p1_fixed_2crash_2gc",
    "PoneAsWritten": "p1_as_written",
    "PoneAsWrittenCommitted": "p1_as_written_committed",
    "PoneIdxFsync": "p1_bug_index_before_fsync",
    "PoneIdxFsyncCommitted": "p1_bug_index_before_fsync_committed",
    "PoneGcEpoch": "p1_bug_gc_ignore_epoch",
    "PoneSanity": "p1_sanity_reachable",
    "PthreeFixed": "p3_fixed",
    "PthreeTwoFail": "p3_fixed_2fail",
    "PthreeTwoFailLong": "p3_fixed_2fail_T16",
    "PthreeAsWritten": "p3_as_written",
    "PthreeWeak": "p3_as_written_weak",
    "PthreeSanity": "p3_sanity_reachable",
    "PthreeRenewLate": "p3_bug_renew_late",
    "PthreeRemoveEarly": "p3_bug_remove_old_ca_early",
    "PthreeSignEarly": "p3_bug_sign_before_bundle",
}


def _tlc(d, name):
    (r,) = [x for x in d if x["name"] == name]
    return r


for _n, _name in _TLC.items():
    _reg(
        f"tlc{_n}States",
        _T,
        lambda d, n=_name: _tlc(d, n)["distinct_states"],
        "{:,}",
        f"{_name} distinct states",
    )
    _reg(f"tlc{_n}Depth", _T, lambda d, n=_name: _tlc(d, n)["depth"], "{:d}", f"{_name} depth")
    _reg(f"tlc{_n}Trace", _T, lambda d, n=_name: _tlc(d, n)["trace_len"], "{:d}", f"{_name} trace length")
_reg("tlcModels", _T, lambda d: len(d), "{:d}", "TLC models run")
_reg(
    "tlcAsExpected",
    _T,
    lambda d: sum(bool(x["as_expected"]) for x in d),
    "{:d}",
    "models with expected outcome",
)
_reg("tlcPassing", _T, lambda d: sum(x["outcome"] == "pass" for x in d), "{:d}", "models that pass")
_reg(
    "tlcViolating",
    _T,
    lambda d: sum(x["outcome"] == "violation" for x in d),
    "{:d}",
    "models with counterexample",
)
_reg("tlcMaxWall", _T, lambda d: max(x["wall_s"] for x in d), "{:.0f}", "longest TLC run (s)")

# ---------------------------------------------------------------- M6: media
_M = "m6/summary.json"
_PROF = ("clean", "lossy", "bwdrop", "jitter", "overload")
_MMET = {
    "Ga": ("glass_to_answer_mean_s", "{:.2f}"),
    "Vmaf": ("vmaf_mean", "{:.1f}"),
    "Stall": ("stall_time_s", "{:.1f}"),
}
for _p in _PROF:
    for _pol, _pn in (("independent", "Indep"), ("joint", "Joint")):
        for _mn, (_mk, _fmt) in _MMET.items():
            _reg(
                f"media{_p.capitalize()}{_mn}{_pn}",
                _M,
                lambda d, c=f"{_p}-{_pol}", k=_mk: _mean(d["conditions"][c][k]),
                _fmt,
                f"{_p}-{_pol} {_mk}",
            )
for _mn, (_mk, _fmt) in _MMET.items():
    _reg(
        f"mediaOverload{_mn}Block",
        _M,
        lambda d, k=_mk: _mean(d["conditions"]["overload-blocking"][k]),
        _fmt,
        f"overload-blocking {_mk}",
    )
_reg(
    "mediaBlockCorrect",
    _M,
    lambda d: _mean(d["conditions"]["overload-blocking"]["answer_correct_frac"]),
    "{:.0f}",
    "blocking answer correctness",
)


def _mt(key, field):
    return lambda d: d["tests"][key][field]


_reg("mediaBwdropP", _M, _mt("bwdrop:glass_to_answer_mean_s", "p_welch"), "{:.3f}", "bwdrop g2a Welch p")
_reg("mediaBwdropHolm", _M, _mt("bwdrop:glass_to_answer_mean_s", "p_holm"), "{:.2f}", "bwdrop g2a Holm p")
_reg("mediaBwdropG", _M, _mt("bwdrop:glass_to_answer_mean_s", "hedges_g"), "{:.1f}", "bwdrop g2a Hedges g")
_reg("mediaOverP", _M, _mt("overload:glass_to_answer_mean_s", "p_welch"), "{:.3f}", "overload g2a Welch p")
_reg("mediaOverHolm", _M, _mt("overload:glass_to_answer_mean_s", "p_holm"), "{:.2f}", "overload g2a Holm p")
_reg(
    "mediaOverDropJoint", _M, _mt("overload:dropped_backpressure", "mean_a"), "{:.1f}", "overload drops joint"
)
_reg(
    "mediaOverDropIndep", _M, _mt("overload:dropped_backpressure", "mean_b"), "{:.0f}", "overload drops indep"
)
_reg(
    "mediaOverDropHolm", _M, _mt("overload:dropped_backpressure", "p_holm"), "{:.3f}", "overload drops Holm p"
)
_reg(
    "mediaOverGiJoint",
    _M,
    _mt("overload:glass_to_inference_p95_s", "mean_a"),
    "{:.2f}",
    "overload g2i p95 joint",
)
_reg(
    "mediaOverGiIndep",
    _M,
    _mt("overload:glass_to_inference_p95_s", "mean_b"),
    "{:.2f}",
    "overload g2i p95 indep",
)
_reg("mediaTests", _M, lambda d: len(d["tests"]), "{:d}", "per-profile tests in the Holm family")


def _pt(key, field):
    return lambda d: d["pooled_tests"][key][field]


_reg("mediaPoolGaJoint", _M, _pt("glass_to_answer_mean_s", "mean_a"), "{:.2f}", "pooled g2a joint")
_reg("mediaPoolGaIndep", _M, _pt("glass_to_answer_mean_s", "mean_b"), "{:.2f}", "pooled g2a indep")
_reg("mediaPoolGaP", _M, _pt("glass_to_answer_mean_s", "p_paired"), "{:.3f}", "pooled g2a paired p")
_reg("mediaPoolGaHolm", _M, _pt("glass_to_answer_mean_s", "p_paired_holm"), "{:.2f}", "pooled g2a Holm p")
_reg("mediaPoolGiJoint", _M, _pt("glass_to_inference_p95_s", "mean_a"), "{:.2f}", "pooled g2i p95 joint")
_reg("mediaPoolGiIndep", _M, _pt("glass_to_inference_p95_s", "mean_b"), "{:.2f}", "pooled g2i p95 indep")
_reg("mediaPoolGiHolm", _M, _pt("glass_to_inference_p95_s", "p_paired_holm"), "{:.3f}", "pooled g2i Holm p")
_reg("mediaPoolDropJoint", _M, _pt("dropped_backpressure", "mean_a"), "{:.1f}", "pooled drops joint")
_reg("mediaPoolDropIndep", _M, _pt("dropped_backpressure", "mean_b"), "{:.1f}", "pooled drops indep")
_reg("mediaPoolPairs", _M, _pt("glass_to_answer_mean_s", "n_paired"), "{:d}", "pooled pairs")
_reg("mediaTrials", _M, lambda d: sum(c["n"] for c in d["conditions"].values()), "{:d}", "media trials")
_reg("mediaConditions", _M, lambda d: len(d["conditions"]), "{:d}", "media conditions")
_reg("mediaSeeds", _M, lambda d: d["conditions"]["clean-joint"]["n"], "{:d}", "seeds per condition")
_reg("mediaLoadMin", _M, lambda d: d["loadavg_1m"]["min"], "{:.0f}", "host 1-min load, min")
_reg("mediaLoadMax", _M, lambda d: d["loadavg_1m"]["max"], "{:.0f}", "host 1-min load, max")


def _nonblock(d, key, fn):
    return fn(v[key]["mean"] for k, v in d["conditions"].items() if not k.endswith("blocking"))


_reg(
    "mediaCorrectMin",
    _M,
    lambda d: _nonblock(d, "answer_correct_frac", min),
    "{:.2f}",
    "answer correctness, min over non-blocking conditions",
)
_reg(
    "mediaCorrectMax",
    _M,
    lambda d: _nonblock(d, "answer_correct_frac", max),
    "{:.2f}",
    "answer correctness, max over non-blocking conditions",
)
_reg(
    "mediaSkewMs",
    _M,
    lambda d: _nonblock(d, "av_skew_ms_median_abs", max),
    "{:.1f}",
    "median |A/V skew| (ms), max over non-blocking conditions",
)

_CAL = "m6/calibration.json"
for _fps, _w in ((30, "Thirty"), (15, "Fifteen"), (10, "Ten")):
    _reg(
        f"mediaCalFour{_w}",
        _CAL,
        lambda d, f=_fps: d["table"][f"400@{f}"],
        "{:.1f}",
        f"VMAF at 400 kbps, {_fps} fps",
    )

_W = "m6/webrtc_ingest.json"
_reg(
    "rtcGsMean",
    _W,
    lambda d: _mean(d["summary"]["glass_to_server_ms_mean"]),
    "{:.1f}",
    "aiortc glass-to-server mean (ms)",
)
_reg(
    "rtcGsP",
    _W,
    lambda d: _mean(d["summary"]["glass_to_server_ms_p95"]),
    "{:.1f}",
    "aiortc glass-to-server p95 (ms)",
)
_reg(
    "rtcDelivery", _W, lambda d: 100 * _mean(d["summary"]["delivery_frac"]), "{:.1f}", "frames delivered (%)"
)
_reg(
    "rtcTranscode",
    _W,
    lambda d: _mean(d["summary"]["av1_transcode_ms_mean"]),
    "{:.1f}",
    "AV1 transcode (ms/frame)",
)
_reg("rtcKbps", _W, lambda d: _mean(d["summary"]["av1_kbps"]), "{:.0f}", "AV1 transcode bitrate (kbps)")
_reg("rtcRuns", _W, lambda d: len(d["runs"]), "{:d}", "aiortc runs")

# ---------------------------------------------------------------- M7: kernels
for _impl, _w in (("triton", "Triton"), ("metal", "Metal")):
    _src = f"m7/correctness_{_impl}.json"
    _reg(f"kern{_w}Pass", _src, lambda d: d["n_pass"], "{:d}", f"{_impl} cases passing")
    _reg(f"kern{_w}Cases", _src, lambda d: d["n"], "{:d}", f"{_impl} cases")
    for _dt, _dw in (("fp32", "Single"), ("fp16", "Half")):
        _reg(
            f"kern{_w}Err{_dw}",
            _src,
            lambda d, t=_dt: _sci(max(r["max_abs_err"] for r in d["rows"] if r["dtype"] == t)),
            desc=f"{_impl} worst max-abs-err {_dt}",
        )
_reg(
    "kernTolSingle",
    "m7/correctness_triton.json",
    lambda d: _sci(d["tolerances"]["fp32"], 0),
    desc="fp32 tolerance",
)
_reg(
    "kernTolHalf",
    "m7/correctness_triton.json",
    lambda d: _sci(d["tolerances"]["fp16"], 0),
    desc="fp16 tolerance",
)


def _kcase(dev, shape, dtype, impl):
    def f(d):
        (c,) = [x for x in d["cases"] if x["shape"] == shape and x["dtype"] == dtype and x["impl"] == impl]
        return c

    return f


for _shape, _fmt in (("model", "{:.1f}"), ("large", "{:.0f}")):
    for _impl, _w in (
        ("eager_unfused", "Eager"),
        ("sdpa_mask", "Sdpa"),
        ("torch_compile", "Compile"),
        ("jax_jit", "Jax"),
    ):
        _reg(
            f"kernCpu{_shape.capitalize()}{_w}",
            "m7/perf_cpu.json",
            lambda d, g=_kcase("cpu", _shape, "fp32", _impl): _mean(g(d)["latency_ms"]),
            _fmt,
            f"CPU {_shape} fp32 {_impl} latency (ms)",
        )
for _shape, _fmt in (("model", "{:.2f}"), ("large", "{:.1f}")):
    for _dt, _dw in (("fp32", "Single"), ("fp16", "Half")):
        for _impl, _w in (("eager_unfused", "Eager"), ("sdpa_mask", "Sdpa"), ("metal_fused", "Metal")):
            _reg(
                f"kernMps{_shape.capitalize()}{_dw}{_w}",
                "m7/perf_mps.json",
                lambda d, g=_kcase("mps", _shape, _dt, _impl): _mean(g(d)["latency_ms"]),
                _fmt,
                f"MPS {_shape} {_dt} {_impl} latency (ms)",
            )
        _reg(
            f"kernMps{_shape.capitalize()}{_dw}MetalOverSdpa",
            "m7/perf_mps.json",
            lambda d, a=_kcase("mps", _shape, _dt, "metal_fused"), b=_kcase("mps", _shape, _dt, "sdpa_mask"): (
                _mean(a(d)["latency_ms"]) / _mean(b(d)["latency_ms"])
            ),
            "{:.1f}",
            f"MPS {_shape} {_dt} metal/sdpa latency",
        )
_reg(
    "kernMpsModelSingleMetalPeak",
    "m7/perf_mps.json",
    lambda d: 100 * _kcase("mps", "model", "fp32", "metal_fused")(d)["frac_of_peak"],
    "{:.0f}",
    "Metal fused, model fp32: % of measured copy bandwidth (minimum traffic)",
)
_reg("kernMpsPeak", "m7/perf_mps.json", lambda d: _mean(d["peak_copy"]["GBps"]), "{:.0f}", "MPS copy GB/s")
_reg("kernCpuPeak", "m7/perf_cpu.json", lambda d: _mean(d["peak_copy"]["GBps"]), "{:.0f}", "CPU copy GB/s")
_reg("kernCpuThreads", "m7/perf_cpu.json", lambda d: d["threads"], "{:d}", "CPU threads")
_reg(
    "kernPadWaste",
    "m7/perf_cpu.json",
    lambda d: 100 * _kcase("cpu", "model", "fp32", "eager_unfused")(d)["traffic"]["padding_waste"],
    "{:.0f}",
    "padding waste of padded layout, model shape (%)",
)
_reg(
    "kernModelTokens",
    "m7/perf_cpu.json",
    lambda d: _kcase("cpu", "model", "fp32", "eager_unfused")(d)["seqlens_total"],
    "{:,}",
    "tokens, model shape",
)
_reg(
    "kernLargeTokens",
    "m7/perf_cpu.json",
    lambda d: _kcase("cpu", "large", "fp32", "eager_unfused")(d)["seqlens_total"],
    "{:,}",
    "tokens, large shape",
)

_IR = "m7/ir_summary.json"


def _ir(dtype, hints):
    def f(d):
        (c,) = [x for x in d["configs"] if x["config"]["dtype"] == dtype and x["config"]["hints"] == hints]
        return c

    return f


_reg("kernTritonVersion", _IR, lambda d: d["triton"], desc="Triton version used for IR inspection")
for _dt, _dw, _h, _hw in (
    ("fp16", "Half", False, "Plain"),
    ("fp16", "Half", True, "Hints"),
    ("fp32", "Single", True, "Hints"),
):
    _g = _ir(_dt, _h)
    _reg(
        f"kernIr{_dw}{_hw}Smem",
        _IR,
        lambda d, g=_g: g(d)["metadata"]["shared_bytes"] // 1024,
        "{:d}",
        "shared KiB",
    )
    for _op, _ow in (
        ("cp.async", "CpAsync"),
        ("ldmatrix", "Ldmatrix"),
        ("mma", "Mma"),
        ("ld.global", "LdGlobal"),
    ):
        _reg(
            f"kernIr{_dw}{_hw}{_ow}",
            _IR,
            lambda d, g=_g, o=_op: g(d)["ptx_stats"][o],
            "{:d}",
            f"PTX {_op} count",
        )

# ---------------------------------------------------------------- M8: post-training
_PT = {"Lo": "m8/posttrain/summary.json", "Hi": "m8/posttrain/budget1.6e13/summary.json"}
_PTM = {"Warm": "warm", "Sft": "sft", "Dpo": "dpo", "Grpo": "grpo", "GrpoRm": "grpo_rm"}
for _b, _src in _PT.items():
    for _mn, _mk in _PTM.items():
        _reg(
            f"pt{_b}{_mn}Em",
            _src,
            lambda d, m=_mk: _mean(d["table"][m]["exact"]),
            "{:.3f}",
            f"{_mk} exact match",
        )
        _reg(
            f"pt{_b}{_mn}Tool",
            _src,
            lambda d, m=_mk: _mean(d["table"][m]["tool_ok"]),
            "{:.3f}",
            f"{_mk} tool_ok",
        )
    _reg(
        f"pt{_b}Budget",
        _src,
        lambda d: _sci(_mean(d["table"]["sft"]["flops"])),
        desc="post-warm-start FLOPs (SFT used)",
    )
    _reg(
        f"pt{_b}GrpoRmP",
        _src,
        lambda d: d["tests"]["grpo_vs_grpo_rm"]["p_welch"],
        "{:.3f}",
        "Welch p GRPO vs GRPO-RM",
    )
_reg(
    "ptLoSftPmin",
    _PT["Lo"],
    lambda d: min(v["p_welch"] for k, v in d["tests"].items() if k.endswith("_vs_sft")),
    "{:.3f}",
    "min Welch p, others vs SFT, 4e12",
)
_reg(
    "ptLoSftPmax",
    _PT["Lo"],
    lambda d: max(v["p_welch"] for k, v in d["tests"].items() if k.endswith("_vs_sft")),
    "{:.3f}",
    "max Welch p, others vs SFT, 4e12",
)
_reg("ptWarmFlops", _PT["Lo"], lambda d: _sci(_mean(d["table"]["warm"]["flops"])), desc="warm-start FLOPs")
_reg(
    "ptLoSftCiLo",
    _PT["Lo"],
    lambda d: d["table"]["sft"]["exact"]["ci95_lo"],
    "{:.2f}",
    "SFT EM CI low, 4e12",
)
_reg(
    "ptLoSftCiHi",
    _PT["Lo"],
    lambda d: d["table"]["sft"]["exact"]["ci95_hi"],
    "{:.2f}",
    "SFT EM CI high, 4e12",
)
_reg(
    "ptHiSftPbound",
    _PT["Hi"],
    lambda d: _sci(max(v["p_welch"] for k, v in d["tests"].items() if k.endswith("_vs_sft")), 0),
    desc="largest Welch p vs SFT at 1.6e13 (sci)",
)
_reg("ptRmAcc", _PT["Lo"], lambda d: _mean(d["reward_model"]["accuracy"]), "{:.3f}", "RM held-out accuracy")
_reg("ptRmAuc", _PT["Lo"], lambda d: _mean(d["reward_model"]["auc"]), "{:.3f}", "RM held-out AUC")
_reg("ptRmPos", _PT["Lo"], lambda d: _mean(d["reward_model"]["pos_rate"]), "{:.2f}", "RM positive rate")
_reg(
    "ptHiGrpoRmToolCiLo",
    _PT["Hi"],
    lambda d: d["table"]["grpo_rm"]["tool_ok"]["ci95_lo"],
    "{:.2f}",
    "GRPO-RM tool_ok CI low, 1.6e13",
)
_reg("ptSeeds", _PT["Lo"], lambda d: d["n_seeds"], "{:d}", "post-training seeds")


def _avg(x):
    return sum(x) / len(x)


for _i, _s in enumerate(SEEDS):
    _reg(
        f"ptLoSftSeed{_s}",
        f"m8/posttrain/seed{_i}.json",
        lambda d: d["sft"]["eval"]["exact"],
        "{:.2f}",
        f"SFT EM seed {_i}, 4e12",
    )
    _src = f"m8/posttrain/budget1.6e13/seed{_i}.json"
    _reg(
        f"ptRmRewFirst{_s}",
        _src,
        lambda d: _avg(d["grpo_rm"]["mean_train_reward_first"]),
        "{:.2f}",
        f"GRPO-RM train reward, first batches, seed {_i}",
    )
    _reg(
        f"ptRmRewLast{_s}",
        _src,
        lambda d: _avg(d["grpo_rm"]["mean_train_reward_last"]),
        "{:.2f}",
        f"GRPO-RM train reward, last batches, seed {_i}",
    )
    _reg(
        f"ptRmTool{_s}",
        _src,
        lambda d: d["grpo_rm"]["eval"]["tool_ok"],
        "{:.2f}",
        f"GRPO-RM tool_ok seed {_i}",
    )
    _reg(f"ptRmEm{_s}", _src, lambda d: d["grpo_rm"]["eval"]["exact"], "{:.2f}", f"GRPO-RM EM seed {_i}")
    _reg(
        f"ptGrpoRewLast{_s}",
        _src,
        lambda d: _avg(d["grpo"]["mean_train_reward_last"]),
        "{:.2f}",
        f"GRPO train reward, last batches, seed {_i}",
    )
_reg("ptParams", "m8/posttrain/seed0.json", lambda d: d["warm"]["n_params"], "{:,}", "model parameters")

# ---------------------------------------------------------------- M8: DP-SGD + MIA
_P = "m8/privacy/summary.json"
_EPS = {
    "None": "inf",
    "Clip": "clip",
    "TenTwentyFour": "1024",
    "TwoFiftySix": "256",
    "SixtyFour": "64",
    "Sixteen": "16",
    "Four": "4",
    "One": "1",
}
for _en, _ek in _EPS.items():
    _reg(f"dp{_en}Em", _P, lambda d, e=_ek: _mean(d["table"][e]["utility_exact"]), "{:.3f}", f"eps={_ek} EM")
    _reg(
        f"dp{_en}LossAuc",
        _P,
        lambda d, e=_ek: _mean(d["table"][e]["mia_loss_auc"]),
        "{:.2f}",
        f"eps={_ek} loss AUC",
    )
    _reg(
        f"dp{_en}LiraAuc",
        _P,
        lambda d, e=_ek: _mean(d["table"][e]["v2_lira_globalvar_auc"]),
        "{:.2f}",
        f"eps={_ek} offline LiRA (v2 shadows, global var) AUC",
    )
    _reg(
        f"dp{_en}Tpr",
        _P,
        lambda d, e=_ek: _mean(d["table"][e]["v2_lira_globalvar_tpr@1%fpr"]),
        "{:.3f}",
        f"eps={_ek} LiRA TPR at 1% FPR",
    )
    if _ek not in ("inf", "clip"):
        _reg(
            f"dp{_en}Rdp",
            _P,
            lambda d, e=_ek: _mean(d["table"][e]["epsilon_rdp"]),
            "{:.0f}",
            f"eps={_ek} RDP",
        )
        _reg(
            f"dp{_en}Prv",
            _P,
            lambda d, e=_ek: _mean(d["table"][e]["epsilon_prv"]),
            "{:.1f}",
            f"eps={_ek} PRV",
        )
        _reg(
            f"dp{_en}Sigma",
            _P,
            lambda d, e=_ek: d["table"][e]["noise_multiplier"],
            "{:.2f}",
            f"eps={_ek} sigma",
        )
_reg("dpSteps", _P, lambda d: d["table"]["1"]["steps"], "{:d}", "DP steps")
_reg("dpRate", _P, lambda d: d["table"]["1"]["sample_rate"], "{:.3f}", "Poisson sample rate")
_reg("dpBatch", _P, lambda d: d["hp"]["batch"], "{:d}", "expected batch")
_reg("dpEpochs", _P, lambda d: d["hp"]["epochs"], "{:d}", "epochs")
_reg("dpSeeds", _P, lambda d: d["table"]["1"]["n_seeds"], "{:d}", "DP seeds")
_reg(
    "dpBaseEm", "m8/privacy/base_eval.json", lambda d: d["utility"]["exact"], "{:.3f}", "public base model EM"
)

# ---------------------------------------------------------------- M8: federated
_F = "m8/federated/summary.json"
for _mn, _mk in (
    ("Avg", "fedavg"),
    ("Prox", "fedprox"),
    ("Central", "central"),
    ("LocalTor", "local_TOR"),
    ("LocalMtl", "local_MTL"),
    ("LocalUs", "local_US"),
):
    _reg(
        f"fed{_mn}Em", _F, lambda d, m=_mk: _mean(d["table"][m]["global_exact"]), "{:.3f}", f"{_mk} global EM"
    )
_reg(
    "fedAvgOverCentral",
    _F,
    lambda d: (
        100 * _mean(d["table"]["fedavg"]["global_exact"]) / _mean(d["table"]["central"]["global_exact"])
    ),
    "{:.0f}",
    "FedAvg EM as % of centralized",
)
_reg(
    "fedCentralP",
    _F,
    lambda d: _sci(d["tests"]["fedavg_vs_central"]["p_welch"]),
    desc="Welch p FedAvg vs central",
)
_reg(
    "fedProxP",
    _F,
    lambda d: d["tests"]["fedprox_vs_fedavg"]["p_welch"],
    "{:.2f}",
    "Welch p FedProx vs FedAvg",
)
_reg("fedRoundMB", _F, lambda d: d["comm"]["bytes_per_round"] / 1e6, "{:.1f}", "MB per round (accounted)")
_reg("fedTotalMB", _F, lambda d: d["comm"]["total_bytes"] / 1e6, "{:.0f}", "MB total (accounted)")
_reg("fedRounds", _F, lambda d: d["hp"]["rounds"], "{:d}", "rounds")
_reg("fedLocalSteps", _F, lambda d: d["hp"]["local_steps"], "{:d}", "local steps per round")
_reg("fedMu", _F, lambda d: d["hp"]["fedprox_mu"], "{:.2f}", "FedProx mu")
_reg(
    "fedSweepOneEm",
    _F,
    lambda d: _mean(d["comm_sweep"]["r80_l1"]["global_exact"]),
    "{:.3f}",
    "80x1 FedAvg EM",
)
_reg(
    "fedSweepOneGB", _F, lambda d: d["comm_sweep"]["r80_l1"]["total_bytes"] / 1e9, "{:.1f}", "80x1 bytes (GB)"
)
_reg(
    "fedRawMB",
    "m8/federated/seed0.json",
    lambda d: d["central"]["comm"]["raw_data_bytes_pooled"] / 1e6,
    "{:.2f}",
    "raw episodes pooled for centralized training, seed 0 (MB)",
)
