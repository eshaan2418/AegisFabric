"""Seeded userspace UDP impairment relay (netem-like): delay, jitter, Gilbert-Elliott loss, rate limit.

This is a *userspace* emulation running on asyncio over 127.0.0.1 UDP. It is not kernel netem.
Timer granularity is ~1 ms. Every random draw comes from a seeded `numpy` Generator, so loss and jitter
sequences are reproducible for a given seed and packet order.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinkProfile:
    name: str
    delay_ms: float = 20.0
    jitter_ms: float = 0.0
    # Gilbert-Elliott: P(good->bad), P(bad->good), loss prob in bad state, loss prob in good state
    ge_p: float = 0.0
    ge_r: float = 1.0
    ge_loss_bad: float = 1.0
    ge_loss_good: float = 0.0
    rate_kbps: float | None = None
    queue_ms: float = 150.0  # drop-tail queue limit (as delay at the current rate)
    rate_schedule: list[tuple[float, float]] = field(default_factory=list)  # (t_s, rate_kbps)

    def mean_loss(self) -> float:
        if self.ge_p == 0:
            return self.ge_loss_good
        pi_bad = self.ge_p / (self.ge_p + self.ge_r)
        return pi_bad * self.ge_loss_bad + (1 - pi_bad) * self.ge_loss_good


PROFILES = {
    "clean": LinkProfile("clean", delay_ms=30, jitter_ms=3, rate_kbps=1500),
    # ~3% mean loss in bursts (avg burst ~2 pkts), 15 ms jitter
    "lossy": LinkProfile(
        "lossy", delay_ms=40, jitter_ms=15, ge_p=0.0155, ge_r=0.5, ge_loss_bad=1.0, rate_kbps=1500
    ),
    # bandwidth collapse 1200 -> 250 kbps at t=8 s, recover at t=16 s
    "bwdrop": LinkProfile(
        "bwdrop",
        delay_ms=30,
        jitter_ms=5,
        ge_loss_good=0.005,
        rate_kbps=1200,
        rate_schedule=[(8.0, 250.0), (16.0, 1200.0)],
    ),
    # high delay variation (normal, sd 40 ms; can reorder) with 1% random loss
    "jitter": LinkProfile("jitter", delay_ms=50, jitter_ms=40, ge_loss_good=0.01, rate_kbps=1500),
}
REVERSE = LinkProfile("reverse", delay_ms=30, jitter_ms=2)


class _Dir:
    """One impaired direction."""

    def __init__(self, prof: LinkProfile, rng: np.random.Generator, t0: float):
        self.p = prof
        self.rng = rng
        self.t0 = t0
        self.bad = False
        self.link_free = 0.0
        self.stats = {"in": 0, "lost": 0, "qdrop": 0, "out_bytes": 0}

    def rate(self, now: float) -> float | None:
        r = self.p.rate_kbps
        for ts, rk in self.p.rate_schedule:
            if now - self.t0 >= ts:
                r = rk
        return r

    def schedule(self, size: int, now: float) -> float | None:
        """Return delivery time for a packet arriving `now`, or None if dropped."""
        self.stats["in"] += 1
        p = self.p
        if p.ge_p > 0:
            self.bad = (self.rng.random() >= p.ge_r) if self.bad else (self.rng.random() < p.ge_p)
        loss_p = p.ge_loss_bad if self.bad else p.ge_loss_good
        if loss_p > 0 and self.rng.random() < loss_p:
            self.stats["lost"] += 1
            return None
        rate = self.rate(now)
        depart = now
        if rate:
            start = max(now, self.link_free)
            if start - now > p.queue_ms / 1000:
                self.stats["qdrop"] += 1
                return None
            depart = start + size * 8 / (rate * 1000)
            self.link_free = depart
        jit = float(self.rng.normal(0, p.jitter_ms)) if p.jitter_ms > 0 else 0.0
        self.stats["out_bytes"] += size
        return depart + max(0.0, p.delay_ms + jit) / 1000


class Relay:
    """Bidirectional relay: packets from `a` go to `b` impaired by `fwd`; packets from `b` go back via `rev`."""

    def __init__(self, fwd: LinkProfile, rev: LinkProfile, seed: int):
        self.fwd_p, self.rev_p = fwd, rev
        self.seed = seed
        self.a_addr: tuple | None = None
        self.b_addr: tuple | None = None

    async def start(self, b_addr: tuple) -> tuple:
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        ss = np.random.SeedSequence(self.seed).spawn(2)
        self.fwd = _Dir(self.fwd_p, np.random.default_rng(ss[0]), t0)
        self.rev = _Dir(self.rev_p, np.random.default_rng(ss[1]), t0)
        self.b_addr = b_addr
        relay = self

        class Proto(asyncio.DatagramProtocol):
            def connection_made(self, tr):
                self.tr = tr

            def datagram_received(self, data, addr):
                now = time.monotonic()
                if addr == relay.b_addr:
                    d, dst = relay.rev, relay.a_addr
                else:
                    relay.a_addr = addr
                    d, dst = relay.fwd, relay.b_addr
                if dst is None:
                    return
                at = d.schedule(len(data), now)
                if at is not None:
                    loop.call_at(loop.time() + (at - now), self.tr.sendto, data, dst)

        self.transport, _ = await loop.create_datagram_endpoint(Proto, local_addr=("127.0.0.1", 0))
        return self.transport.get_extra_info("sockname")

    def close(self) -> None:
        self.transport.close()

    def stats(self) -> dict:
        return {"fwd": self.fwd.stats, "rev": self.rev.stats}
