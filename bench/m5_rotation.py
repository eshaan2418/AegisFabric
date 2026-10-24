"""M5: control-plane security under live training load.

Configs (each a 6-rank DDP job on the WAN lab, nominal network, static routing; every rank reports at 1 Hz
to the controller over the out-of-band management network):
  none        no control channel (reference step time)
  plaintext   gRPC without TLS
  mtls        mTLS (TLS 1.3, SPIFFE authz), cert TTL 3600 s (no rotation during the run)
  mtls-rot    mTLS with TTL 10 s (renewal every ~5 s per workload) plus a two-phase CA rotation mid-run
Measured: failed steps / rank failures, RPC errors, RPC latency, handshake latency (new channel → ready),
renewal (issue) latency, step-time overhead vs `none`.

    python -m bench.m5_rotation --trials 5 --steps 60
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from bench.meta import write_result
from bench.stats import summarize, welch
from controller.planner import DEFAULT_PLACEMENT
from netlab.lab import MGMT_NET, default_lab
from security.ca import spiffe_id, write_atomic
from security.service import ControlPlane
from train.launch import Job, Slot, read_rank_log

CONFIGS = ["none", "plaintext", "mtls", "mtls-rot"]


def run_one(lab, cfg: str, trial: int, steps: int, tmp: Path) -> dict:
    out = tmp / f"{cfg}_{trial}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    used: dict = {}
    slots = []
    for s in DEFAULT_PLACEMENT:
        used[s] = used.get(s, 0) + 1
        h = f"{s}-c{used[s]}"
        slots.append(Slot(h, lab.topo.host_ip(h), "eth0"))
    args = [
        "--mode",
        "ddp",
        "--preset",
        "small",
        "--batch",
        "16",
        "--steps",
        str(steps),
        "--eval-every",
        "0",
        "--threads",
        "2",
        "--job",
        f"m5-{cfg}-{trial}",
    ]
    cp = None
    rot_log = []
    ttl = 10.0 if cfg == "mtls-rot" else 3600.0
    if cfg != "none":
        cp = ControlPlane(
            bind=f"{MGMT_NET}.1:7500",
            ttl_s=ttl,
            skew_s=0.5,
            server_ips=[f"{MGMT_NET}.1"],
            secure=cfg != "plaintext",
        )
        cp.start()
        args += ["--control-grpc", f"{MGMT_NET}.1:7500", "--cert-ttl", str(ttl)]
        if cfg == "plaintext":
            args += ["--control-insecure"]
        else:
            for r, (site, sl) in enumerate(zip(DEFAULT_PLACEMENT, slots, strict=True)):
                d = out / f"id{r}"
                d.mkdir()
                # bootstrap SVID (stand-in for SPIRE attestation at start); lifetime covers worker start-up,
                # after which the workload renews onto the short-lived rotation schedule
                iss = cp.ca.issue(spiffe_id(site, "trainer", str(r)), max(ttl, 180.0))
                write_atomic(d / "cert.pem", iss.cert_pem)
                write_atomic(d / "key.pem", iss.key_pem)
                write_atomic(d / "bundle.pem", cp.bundle.pem())
                del sl
            args += ["--identity-dir", str(out / "id{rank}")]
    job = Job(slots, args, out, port=30500 + trial * 10 + CONFIGS.index(cfg))
    job.start()
    if cfg == "mtls-rot":

        def rotate():
            # wait for training to start, then rotate the CA in two phases (P3)
            time.sleep(25)
            rot_log.append({"wall": time.time(), "event": "ca_phase1"})
            cp.begin_ca_rotation()
            time.sleep(ttl + 2 * 0.5 + 1)
            old = cp.commit_ca_rotation()
            rot_log.append({"wall": time.time(), "event": "ca_phase2"})
            time.sleep(ttl + 2 * 0.5 + 1)
            cp.retire_ca(old)
            rot_log.append({"wall": time.time(), "event": "ca_retired"})

        threading.Thread(target=rotate, daemon=True).start()
    codes = job.wait(1800)
    res = {"cfg": cfg, "trial": trial, "exit_codes": codes}
    steps_r0 = [r for r in read_rank_log(out, 0) if r.get("event") == "step"]
    res["steps_completed_rank0"] = len(steps_r0)
    st = np.array([r["step_s"] for r in steps_r0[3:]])
    res["step_mean_s"] = float(st.mean()) if len(st) else None
    res["step_p99_s"] = float(np.percentile(st, 99)) if len(st) else None
    errors, lat, hs, ren, ren_fail = [], [], [], [], 0
    failed_ranks = 0
    for r in range(6):
        recs = read_rank_log(out, r)
        if not any(x.get("event") == "end" for x in recs):
            failed_ranks += 1
        cc = next((x for x in recs if x.get("event") == "control_channel"), None)
        if cc:
            errors += cc["errors"]
            lat += [x[1] for x in cc["rpc_lat"]]
            hs += cc["handshakes"]
            ren += [x["issue_s"] for x in cc["renewals"]]
            ren_fail += cc["renew_failures"]
    res.update(
        {
            "failed_ranks": failed_ranks,
            "rpc_errors": len(errors),
            "error_samples": errors[:5],
            "rpc_lat_ms": summarize(np.array(lat) * 1e3, with_p99=True).as_dict() if lat else None,
            "n_rpcs": len(lat),
            "handshake_ms": (np.array(hs) * 1e3).tolist(),
            "renewal_ms": (np.array(ren) * 1e3).tolist(),
            "renew_failures": ren_fail,
            "denied": cp.authz.denied if cp and cp.secure else 0,
            "ca_rotation": rot_log,
            "server_reports": cp.servicer.reports if cp else 0,
        }
    )
    if cp:
        cp.stop()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="results/m5")
    a = ap.parse_args()
    lab = default_lab()
    lab.build()
    lab.start_frr()
    lab.wait_converged()
    tmp = Path("/tmp/aegis_m5")
    rows = []
    rng = np.random.default_rng(5)
    for t in range(a.trials):
        for k in rng.permutation(len(CONFIGS)):  # interleave configs, random order per trial
            r = run_one(lab, CONFIGS[k], t, a.steps, tmp)
            print({x: r[x] for x in ("cfg", "trial", "failed_ranks", "rpc_errors", "step_mean_s", "n_rpcs")})
            rows.append(r)
    lab.teardown()
    summ = {}
    base = [r["step_mean_s"] for r in rows if r["cfg"] == "none"]
    for c in CONFIGS:
        rs = [r for r in rows if r["cfg"] == c]
        hs = [h for r in rs for h in r["handshake_ms"]]
        ren = [h for r in rs for h in r["renewal_ms"]]
        summ[c] = {
            "step_mean_s": summarize([r["step_mean_s"] for r in rs]).as_dict(),
            "failed_ranks_total": sum(r["failed_ranks"] for r in rs),
            "rpc_errors_total": sum(r["rpc_errors"] for r in rs),
            "rpcs_total": sum(r["n_rpcs"] for r in rs),
            "renewals_total": len(ren),
            "handshake_ms": summarize(hs, with_p99=True).as_dict() if hs else None,
            "renewal_ms": summarize(ren, with_p99=True).as_dict() if ren else None,
            "rpc_lat_ms_mean": summarize([r["rpc_lat_ms"]["mean"] for r in rs if r["rpc_lat_ms"]]).as_dict()
            if any(r["rpc_lat_ms"] for r in rs)
            else None,
            "vs_none": welch([r["step_mean_s"] for r in rs], base) if c != "none" else None,
        }
    write_result(Path(a.out) / "rotation.json", {"summary": summ, "runs": rows}, config=vars(a), seed=5)


if __name__ == "__main__":
    main()
