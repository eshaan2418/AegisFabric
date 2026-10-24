"""Seeded background (other-tenant) traffic: constant-bit-rate UDP flows between site `bg` hosts.

Background prefixes follow a fixed routing policy that the training controller does not control (it is
another tenant's traffic). UDP CBR does not back off, so it removes a known amount of capacity from the
links it crosses, which makes the experiments deterministic given the seed.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from netlab import te

# Fixed background routing policy (not under the controller's authority).
BG_POLICY = {
    ("tor", "mtl"): "a",
    ("mtl", "tor"): "a",
    ("tor", "us"): "b",
    ("us", "tor"): "b",
    ("mtl", "us"): "b",
    ("us", "mtl"): "b",
}


@dataclasses.dataclass
class BgFlow:
    start_s: float
    duration_s: float
    src: str
    dst: str
    rate_mbit: float


def schedule(
    rng: np.random.Generator, horizon_s: float, mean_on_s: float = 40.0, rates=(40, 80, 150), n_flows: int = 6
) -> list[BgFlow]:
    pairs = list(BG_POLICY)
    out = []
    for _ in range(n_flows):
        src, dst = pairs[int(rng.integers(0, len(pairs)))]
        start = float(rng.uniform(0, max(1.0, horizon_s - 20)))
        dur = float(min(horizon_s - start, rng.exponential(mean_on_s) + 10))
        out.append(BgFlow(start, dur, src, dst, float(rng.choice(rates))))
    return sorted(out, key=lambda f: f.start_s)


def apply_policy(lab, policy=BG_POLICY) -> None:
    decisions = {(s, d, "bg"): p for (s, d), p in policy.items()}
    te.apply(lab, decisions)


class Background:
    def __init__(self, lab, flows: list[BgFlow], log: Path, base_port: int = 6000):
        self.lab, self.flows, self.log = lab, flows, log
        self.base_port = base_port
        self.procs: list[subprocess.Popen] = []

    def start(self) -> None:
        self.t0 = time.monotonic()
        for k, f in enumerate(self.flows):
            port = self.base_port + k
            self.procs.append(
                subprocess.Popen(
                    ["ip", "netns", "exec", f"{f.dst}-bg", "iperf3", "-s", "-1", "-p", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        time.sleep(0.5)
        import threading

        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self) -> None:
        for k, f in enumerate(self.flows):
            d = self.t0 + f.start_s - time.monotonic()
            if d > 0:
                time.sleep(d)
            dst_ip = self.lab.topo.host_ip(f"{f.dst}-bg")
            p = subprocess.Popen(
                [
                    "ip",
                    "netns",
                    "exec",
                    f"{f.src}-bg",
                    "iperf3",
                    "-c",
                    dst_ip,
                    "-p",
                    str(self.base_port + k),
                    "-u",
                    "-b",
                    f"{f.rate_mbit}M",
                    "-l",
                    "1200",
                    "-t",
                    str(max(1, int(f.duration_s))),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs.append(p)
            with open(self.log, "a") as fh:
                fh.write(
                    json.dumps({"wall": time.time(), "event": "bg_start", **dataclasses.asdict(f)}) + "\n"
                )

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        subprocess.run("pkill -9 iperf3", shell=True, check=False)
