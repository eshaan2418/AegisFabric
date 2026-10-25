"""Rate estimation and adaptation policies.

* `LossBasedRate`: sender-side, GCC-style loss controller driven by RTCP RR fraction-lost.
* `DelayBasedEstimator`: receiver-side, GCC-style delay-gradient over-use detector. Its estimate is
  sent back as REMB.
* `IndependentPolicy`: each subsystem adapts on its own. ABR picks the bitrate rung at a fixed 30 fps,
  the frame selector runs at a fixed max rate, and the inference queue is FIFO.
* `JointPolicy`: one controller picks (bitrate rung, encoder fps, selection rate, inference policy) under a
  bandwidth estimate and the measured inference service rate. Encoder fps comes from an offline
  calibration table Q(bitrate, fps) → VMAF (results/m6/calibration.json, measured by bench/m6_calibrate.py).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

LADDER_KBPS = [100, 200, 400, 700, 1000]
FPS_CHOICES = [30, 15, 10]


class LossBasedRate:
    def __init__(self, start_kbps: float, lo: float = 80, hi: float = 1200):
        self.rate = start_kbps
        self.lo, self.hi = lo, hi

    def on_loss(self, frac_lost: float) -> float:
        if frac_lost > 0.10:
            self.rate *= 1 - 0.5 * frac_lost
        elif frac_lost < 0.02:
            self.rate *= 1.05
        self.rate = min(max(self.rate, self.lo), self.hi)
        return self.rate


class DelayBasedEstimator:
    """Receiver: tracks one-way-delay trend and incoming rate; returns a REMB estimate (bps)."""

    def __init__(self, window_s: float = 0.5, overuse_ms: float = 15.0):
        self.window = window_s
        self.overuse = overuse_ms / 1000
        self.arrivals: collections.deque[tuple[float, int, float]] = collections.deque()  # (t, bytes, owd)
        self.estimate = 600_000.0
        self.min_owd = float("inf")
        self.state = "normal"
        self._over_count = 0
        self.bytes: collections.deque[tuple[float, int]] = collections.deque()
        self.trace: list[tuple[float, float]] = []

    def on_packet(self, t: float, size: int, owd: float | None) -> None:
        """`owd` is given only for the first packet of each frame (excludes intra-frame burst serialisation)."""
        self.bytes.append((t, size))
        while self.bytes and t - self.bytes[0][0] > self.window:
            self.bytes.popleft()
        if owd is None:
            return
        self.arrivals.append((t, size, owd))
        self.trace.append((t, owd))
        self.min_owd = min(self.min_owd, owd)
        while self.arrivals and t - self.arrivals[0][0] > self.window:
            self.arrivals.popleft()

    def update(self, t: float) -> float:
        """Trendline over-use detector (GCC-like). Over-use requires a rising one-way-delay trend
        (>15 ms over the window) AND standing queueing delay (>25 ms), on two consecutive updates."""
        if len(self.arrivals) < 6:
            return self.estimate
        arr = np.array(self.arrivals)
        span = max(self.bytes[-1][0] - self.bytes[0][0], 1e-3)
        rx = sum(b for _, b in self.bytes) * 8 / span
        slope = float(np.polyfit(arr[:, 0] - arr[0, 0], arr[:, 2], 1)[0])
        tail = arr[-max(len(arr) // 3, 2) :, 2]
        queueing = float(np.median(tail)) - self.min_owd
        over = slope * self.window > self.overuse and queueing > 0.025
        self._over_count = self._over_count + 1 if over else 0
        if self._over_count >= 2:
            self.state = "overuse"
            self.estimate = max(0.85 * rx, 50_000)
        else:
            self.state = "normal"
            self.estimate = min(self.estimate * 1.08, 1.5 * rx + 250_000, 2_000_000)
        return self.estimate


@dataclass
class Decision:
    bitrate_kbps: int
    fps: int
    select_fps: float
    queue_policy: str


def pick_rung(target_kbps: float, current: int, since_change_s: float) -> int:
    cand = max([r for r in LADDER_KBPS if r <= 0.9 * target_kbps] or [LADDER_KBPS[0]])
    if cand < current and target_kbps < 0.95 * current:
        return cand  # down-switch immediately
    if cand > current and since_change_s >= 1.5:
        return min(cand, next(r for r in LADDER_KBPS if r > current))  # step up one rung at a time
    return current


class IndependentPolicy:
    name = "independent"

    def __init__(self, select_fps: float = 10.0):
        self.select_fps = select_fps

    def decide(
        self, target_kbps: float, cur: Decision, since_change_s: float, service_fps: float
    ) -> Decision:
        return Decision(pick_rung(target_kbps, cur.bitrate_kbps, since_change_s), 30, self.select_fps, "fifo")


class JointPolicy:
    name = "joint"

    def __init__(self, quality_table: dict[str, float], select_cap: float = 10.0):
        # quality_table["{kbps}@{fps}"] = mean VMAF from calibration
        self.q = quality_table
        self.select_cap = select_cap

    def best_fps(self, kbps: int) -> int:
        scored = [(self.q.get(f"{kbps}@{f}", -1.0), f) for f in FPS_CHOICES]
        return max(scored)[1]

    def decide(
        self, target_kbps: float, cur: Decision, since_change_s: float, service_fps: float
    ) -> Decision:
        rung = pick_rung(target_kbps, cur.bitrate_kbps, since_change_s)
        fps = self.best_fps(rung)
        if fps != cur.fps and since_change_s < 1.5 and rung == cur.bitrate_kbps:
            fps = cur.fps  # hysteresis: encoder re-init costs a keyframe
        # never select faster than the accelerator can serve (80% utilisation) or than frames arrive
        sel = max(1.0, min(self.select_cap, fps, 0.8 * service_fps))
        return Decision(rung, fps, sel, "latest")
