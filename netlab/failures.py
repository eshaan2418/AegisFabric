"""Seeded, scripted, timestamped failure injection on WAN links.

Failure types (all apply to one inter-site link of one plane):
  link_down      carrier down on one end (peer sees NO-CARRIER) for `duration` s, then restored
  flap           `count` cycles of down `down_s` / up `up_s`
  bw_collapse    tbf rate reduced to `rate_mbit` for `duration` s
  latency_spike  netem delay increased by `extra_ms` for `duration` s
  bgp_reset      hard reset of the eBGP session on that link (`clear ip bgp <peer>`)
  blackhole      silent failure: 100% loss on both directions, carrier stays up (detected by BGP hold timer)
Every action is logged to a JSONL file with wall-clock and monotonic timestamps.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np

from netlab.topology import LinkSpec

FAILURE_TYPES = ("link_down", "blackhole", "flap", "bw_collapse", "latency_spike", "bgp_reset")


@dataclasses.dataclass
class Failure:
    at_s: float
    kind: str
    plane: str
    s1: str
    s2: str
    duration: float = 30.0
    rate_mbit: float = 20.0
    extra_ms: float = 80.0
    count: int = 3
    down_s: float = 2.0
    up_s: float = 4.0


def random_failure(
    rng: np.random.Generator, kind: str, at_s: float, plane: str = "a", pair: tuple[str, str] | None = None
) -> Failure:
    pairs = [("tor", "mtl"), ("tor", "us"), ("mtl", "us")]
    s1, s2 = pair or pairs[int(rng.integers(0, 3))]
    return Failure(
        at_s=at_s,
        kind=kind,
        plane=plane,
        s1=s1,
        s2=s2,
        duration=float(rng.uniform(25, 35)),
        rate_mbit=float(rng.choice([10, 20, 40])),
        extra_ms=float(rng.choice([50, 80, 120])),
    )


class Injector:
    def __init__(self, lab, log_path: Path):
        self.lab = lab
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._threads: list[threading.Thread] = []
        self.t0 = time.monotonic()

    def log(self, **kw) -> None:
        rec = {"wall": time.time(), "mono": time.monotonic() - self.t0, **kw}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def schedule(self, failures: list[Failure]) -> None:
        for f in failures:
            th = threading.Thread(target=self._run, args=(f,), daemon=True)
            th.start()
            self._threads.append(th)

    def join(self, timeout: float | None = None) -> None:
        for th in self._threads:
            th.join(timeout)

    def _sleep_until(self, t: float) -> None:
        d = self.t0 + t - time.monotonic()
        if d > 0:
            time.sleep(d)

    def _run(self, f: Failure) -> None:
        self._sleep_until(f.at_s)
        lk = self.lab.topo.wan_link(f.plane, f.s1, f.s2)
        nominal = dataclasses.replace(lk.spec)
        base = {"kind": f.kind, "plane": f.plane, "s1": f.s1, "s2": f.s2}
        if f.kind == "link_down":
            self.lab.wan_link_state(f.plane, f.s1, f.s2, up=False)
            self.log(event="inject", **base, duration=f.duration)
            time.sleep(f.duration)
            self.lab.wan_link_state(f.plane, f.s1, f.s2, up=True)
            self.log(event="restore", **base)
        elif f.kind == "flap":
            self.log(event="inject", **base, count=f.count, down_s=f.down_s, up_s=f.up_s)
            for _ in range(f.count):
                self.lab.wan_link_state(f.plane, f.s1, f.s2, up=False)
                self.log(event="flap_down", **base)
                time.sleep(f.down_s)
                self.lab.wan_link_state(f.plane, f.s1, f.s2, up=True)
                self.log(event="flap_up", **base)
                time.sleep(f.up_s)
            self.log(event="restore", **base)
        elif f.kind == "bw_collapse":
            self.lab.set_wan(f.plane, f.s1, f.s2, dataclasses.replace(nominal, rate_mbit=f.rate_mbit))
            self.log(event="inject", **base, rate_mbit=f.rate_mbit, duration=f.duration)
            time.sleep(f.duration)
            self.lab.set_wan(f.plane, f.s1, f.s2, nominal)
            self.log(event="restore", **base)
        elif f.kind == "latency_spike":
            spec = LinkSpec(
                nominal.delay_ms + f.extra_ms, nominal.rate_mbit, nominal.jitter_ms, nominal.loss_pct
            )
            self.lab.set_wan(f.plane, f.s1, f.s2, spec)
            self.log(event="inject", **base, extra_ms=f.extra_ms, duration=f.duration)
            time.sleep(f.duration)
            self.lab.set_wan(f.plane, f.s1, f.s2, nominal)
            self.log(event="restore", **base)
        elif f.kind == "blackhole":
            self.lab.set_wan(f.plane, f.s1, f.s2, dataclasses.replace(nominal, loss_pct=100.0))
            self.log(event="inject", **base, duration=f.duration)
            time.sleep(f.duration)
            self.lab.set_wan(f.plane, f.s1, f.s2, nominal)
            self.log(event="restore", **base)
        elif f.kind == "bgp_reset":
            self.lab.bgp_reset(f.plane, f.s1, f.s2)
            self.log(event="inject", **base)
        else:
            raise ValueError(f.kind)
