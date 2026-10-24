"""Seeded experiment scenarios for the WAN training study (M2/M3).

Every failure scenario contains the same failure type twice, once on each plane, on the same site pair:
first on plane B (at ~T1), then on plane A (at ~T2). A method that routes over B and a method that routes
over A therefore each face exactly one failure of their path, which keeps the comparison fair to static
routing (always plane A). Timings and the affected pair are drawn from the seed. Background traffic is a
seeded set of UDP CBR flows that follows a fixed policy not under the controller's authority.
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np

from netlab.background import BgFlow, schedule
from netlab.failures import Failure

PAIRS = [("tor", "mtl"), ("tor", "us"), ("mtl", "us")]


@dataclasses.dataclass
class Scenario:
    name: str
    steps: int
    ckpt_every: int
    failures: list[Failure]
    bg: list[BgFlow]
    horizon_s: float


def make(name: str, seed: int, steps: int = 160, ckpt_every: int = 20, horizon_s: float = 260.0) -> Scenario:
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    rng_bg = np.random.default_rng([seed, 7])  # background identical across scenarios for a given seed
    bg = schedule(rng_bg, horizon_s, n_flows=5)
    fails: list[Failure] = []
    t1 = float(rng.uniform(35, 45))
    t2 = float(rng.uniform(125, 135))
    pair = ("tor", "mtl")  # the only inter-site pair every evaluated placement uses (see DECISIONS D-007)
    dur = float(rng.uniform(35, 45))
    if name == "steady":
        pass
    elif name in ("link_down", "blackhole", "bw_collapse", "latency_spike"):
        for t, plane in ((t1, "b"), (t2, "a")):
            fails.append(
                Failure(
                    at_s=t,
                    kind=name,
                    plane=plane,
                    s1=pair[0],
                    s2=pair[1],
                    duration=dur,
                    rate_mbit=float(rng.choice([10, 20])),
                    extra_ms=float(rng.choice([80, 120])),
                )
            )
    elif name == "flap":
        for t, plane in ((t1, "b"), (t2, "a")):
            fails.append(
                Failure(
                    at_s=t,
                    kind="flap",
                    plane=plane,
                    s1=pair[0],
                    s2=pair[1],
                    count=5,
                    down_s=float(rng.uniform(1.5, 3.0)),
                    up_s=float(rng.uniform(3.0, 6.0)),
                )
            )
    elif name == "bgp_reset":
        for t, plane in ((t1, "b"), (t2, "a")):
            for k in range(3):  # a burst of resets (e.g. a misbehaving peer / maintenance mistake)
                fails.append(Failure(at_s=t + 8 * k, kind="bgp_reset", plane=plane, s1=pair[0], s2=pair[1]))
    elif name == "site_degradation":
        # every WAN link touching one site loses most of its capacity for a long period (e.g. a regional
        # carrier problem). Routing alone cannot fix this; only placement can.
        site = str(rng.choice(["tor", "mtl"]))
        for plane in ("a", "b"):
            for p in PAIRS:
                if site in p:
                    fails.append(
                        Failure(
                            at_s=t1,
                            kind="bw_collapse",
                            plane=plane,
                            s1=p[0],
                            s2=p[1],
                            duration=float(horizon_s),
                            rate_mbit=30.0,
                        )
                    )
    else:
        raise ValueError(name)
    return Scenario(name, steps, ckpt_every, fails, bg, horizon_s)


MAIN = [
    "steady",
    "link_down",
    "blackhole",
    "flap",
    "bw_collapse",
    "latency_spike",
    "bgp_reset",
    "site_degradation",
]
