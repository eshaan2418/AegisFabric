"""M2: control- and data-plane convergence per failure type (static/default routing).

For each failure type and trial: a 5 ms UDP prober runs tor-c3 → mtl-c3 (compute prefix, default path =
plane-A direct tor–mtl), and the failure is injected on that link at t=3 s. Measured:
  outage_s        longest gap in received echoes after injection (data plane)
  fib_change_s    injection → tor-gw FIB next hop for 10.2.1.0/24 changes (control plane → kernel)
  restore_*       the same after the link comes back (link_down / flap)
  bgp_reestablish_s  injection → eBGP session on the link Established again (bgp_reset)
For bw_collapse / latency_spike BGP does not react, so we report RTT and TCP throughput before/during.

Run inside the lab container:  python -m bench.m2_convergence --trials 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from bench.meta import write_result
from bench.stats import summarize
from netlab import probe as P
from netlab.failures import Failure, Injector
from netlab.lab import default_lab, nsexec, run

SRC, DST, DST_IP = "tor-c3", "mtl-c3", "10.2.1.13"


class FibWatch(threading.Thread):
    def __init__(self, ns: str, prefix: str, interval: float = 0.01):
        super().__init__(daemon=True)
        self.ns, self.prefix, self.interval = ns, prefix, interval
        self.changes: list[tuple[float, str]] = []
        self.stop_flag = False

    def run(self) -> None:
        last = None
        while not self.stop_flag:
            out = run(nsexec(self.ns, "ip", "route", "show", self.prefix), capture=True, check=False).strip()
            dev = "to-a" if "to-a" in out else "to-b" if "to-b" in out else "none"
            if "to-a" in out and "to-b" in out:
                dev = "ecmp"
            if dev != last:
                self.changes.append((time.time(), dev))
                last = dev
            time.sleep(self.interval)


def bgp_state(lab, router: str, peer_ip: str) -> str:
    s = lab.bgp_summary(router).get("peers", {}).get(peer_ip, {})
    return s.get("state", "?")


def iperf_mbit(seconds: int = 4) -> float:
    srv = subprocess.Popen(
        nsexec(DST, "iperf3", "-s", "-1", "-p", "5300"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(0.3)
    out = run(
        nsexec(SRC, "iperf3", "-c", DST_IP, "-p", "5300", "-t", str(seconds), "-J"),
        capture=True,
        check=False,
        timeout=seconds + 15,
    )
    srv.kill()
    try:
        return json.loads(out)["end"]["sum_received"]["bits_per_second"] / 1e6
    except (json.JSONDecodeError, KeyError):
        return float("nan")


def ping_ms(n: int = 10) -> float:
    out = run(nsexec(SRC, "ping", "-c", str(n), "-i", "0.1", DST_IP), capture=True, check=False)
    for line in out.splitlines():
        if "rtt" in line:
            return float(line.split("=")[1].split("/")[1])
    return float("nan")


def trial(lab, kind: str, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    lk = lab.topo.wan_link("a", "tor", "mtl")
    peer = lk.b_ip.split("/")[0]
    f = Failure(
        at_s=3.0,
        kind=kind,
        plane="a",
        s1="tor",
        s2="mtl",
        duration=float(rng.uniform(6, 9)),
        rate_mbit=20,
        extra_ms=80,
        count=3,
        down_s=2.0,
        up_s=4.0,
    )
    res: dict = {"kind": kind, "seed": seed}
    if kind in ("bw_collapse", "latency_spike"):
        res["rtt_before_ms"] = ping_ms()
        res["tput_before_mbit"] = iperf_mbit()
        inj = Injector(lab, out_dir / f"inject_{kind}_{seed}.jsonl")
        f.at_s, f.duration = 0.0, 12.0
        inj.schedule([f])
        time.sleep(1.0)
        res["rtt_during_ms"] = ping_ms()
        res["tput_during_mbit"] = iperf_mbit()
        fw = FibWatch("tor-gw", "10.2.1.0/24")
        fw.start()
        inj.join()
        fw.stop_flag = True
        res["fib_changes"] = len(fw.changes) - 1
        return res
    duration = {
        "link_down": 3 + f.duration + 8,
        "blackhole": 3 + f.duration + 8,
        "flap": 3 + 3 * 6 + 6,
        "bgp_reset": 15,
    }[kind]
    probe_out = out_dir / f"probe_{kind}_{seed}.jsonl"
    refl = subprocess.Popen(
        nsexec(DST, "python", "-m", "netlab.probe", "reflect", "--port", "7001"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    snd = subprocess.Popen(
        nsexec(
            SRC,
            "python",
            "-m",
            "netlab.probe",
            "send",
            "--dst",
            DST_IP,
            "--port",
            "7001",
            "--interval",
            "0.005",
            "--duration",
            str(duration),
            "--out",
            str(probe_out),
        )
    )
    fw = FibWatch("tor-gw", "10.2.1.0/24")
    fw.start()
    inj = Injector(lab, out_dir / f"inject_{kind}_{seed}.jsonl")
    t_inj = time.time() + f.at_s
    inj.schedule([f])
    if kind == "bgp_reset":
        time.sleep(f.at_s + 0.05)
        t0 = time.time()
        while time.time() - t0 < 30 and bgp_state(lab, "tor-a", peer) == "Established":
            time.sleep(0.01)
        t_down = time.time()
        while time.time() - t0 < 60 and bgp_state(lab, "tor-a", peer) != "Established":
            time.sleep(0.02)
        res["bgp_down_detect_s"] = t_down - t_inj
        res["bgp_reestablish_s"] = time.time() - t_inj
    snd.wait(timeout=duration + 30)
    refl.kill()
    inj.join()
    fw.stop_flag = True
    recs = [json.loads(x) for x in probe_out.read_text().splitlines() if x.strip()]
    o = P.outage(recs, 0.005, after=t_inj - 0.01)
    res["outage_s"] = o["outage_s"]
    changes = [(t - t_inj, d) for t, d in fw.changes]
    res["fib_changes"] = changes
    first = next((t for t, d in changes if t > 0), None)
    res["fib_change_s"] = first
    if kind in ("link_down", "blackhole"):
        t_restore = f.at_s + f.duration
        back = next((t for t, d in changes if t > t_restore - f.at_s and d == "to-a"), None)
        res["restore_fib_s"] = None if back is None else back - (t_restore - f.at_s)
        # outage around restoration
        o2 = P.outage(recs, 0.005, after=t_inj + f.duration - 0.01)
        res["restore_outage_s"] = o2["outage_s"]
    # data plane loss fraction during the whole probe
    sent = int(duration / 0.005)
    res["loss_frac"] = 1 - len(recs) / max(1, sent)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--kinds", default="link_down,blackhole,flap,bgp_reset,bw_collapse,latency_spike")
    ap.add_argument("--out", default="results/m2")
    a = ap.parse_args()
    out_dir = Path(a.out) / "raw_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = default_lab()
    lab.build()
    lab.start_frr()
    conv = lab.wait_converged()
    rows = []
    for kind in a.kinds.split(","):
        for t in range(a.trials):
            lab.wait_converged()
            time.sleep(2)
            r = trial(lab, kind, 1000 + t, out_dir)
            print(json.dumps({k: v for k, v in r.items() if k != "fib_changes"}))
            rows.append(r)
    summary = {}
    for kind in a.kinds.split(","):
        rs = [r for r in rows if r["kind"] == kind]
        summary[kind] = {}
        for key in (
            "outage_s",
            "fib_change_s",
            "restore_fib_s",
            "restore_outage_s",
            "bgp_down_detect_s",
            "bgp_reestablish_s",
            "rtt_before_ms",
            "rtt_during_ms",
            "tput_before_mbit",
            "tput_during_mbit",
            "loss_frac",
        ):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                summary[kind][key] = summarize(vals).as_dict()
    write_result(
        Path(a.out) / "convergence.json",
        {"initial_convergence_s": conv, "summary": summary, "trials": rows},
        config={
            "trials": a.trials,
            "kinds": a.kinds,
            "probe_interval_s": 0.005,
            "bgp_timers": "keepalive 1 / hold 3",
            "mrai": 0,
        },
        seed=1000,
    )
    lab.teardown()


if __name__ == "__main__":
    main()
