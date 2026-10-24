"""Receiver-side analysis: frame selection, a stub vision model, STT and a single-accelerator scheduler.

* Vision "model": a STUB. A fixed random-weight CNN (compute-cost proxy, padded to a target service
  time) plus a deterministic chart reader that extracts the right-most revealed point of the balance
  chart. Its *accuracy* is the chart-read error versus the same reader applied to the uncompressed
  source frame, so it reflects coding artifacts and staleness, not ML quality.
* STT: whisper-tiny (openai/whisper-tiny via `transformers`) when available, otherwise
  `SimulatedStt`, which returns the oracle text after a latency calibrated as a real-time factor.
  Results record which one ran.
* Scheduler: one worker thread is the shared "accelerator" that runs vision and STT/answer jobs.
  The media path only ever calls non-blocking `submit_*`. Policies:
    - "fifo":    bounded FIFO (cap 8); when full, the new frame is dropped. STT jobs queue FIFO.
    - "latest":  one vision slot, latest wins (older pending frame dropped); STT/answer jobs preempt.
    - "blocking": ablation, a bounded queue whose `put` blocks the caller (media stalls under overload).
"""

from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np


def chart_point(rgb: np.ndarray) -> tuple[float, float]:
    """Right-most revealed chart column and its mean y (normalised 0..1), from the green channel.

    The talking-head disc is excluded by requiring G well above R."""
    g = rgb[..., 1].astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    mask = (g > 120) & (g - r > 60)
    mask[-2:, :] = False  # axis row
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return 0.0, 0.5
    c = cols[-1]
    ys = np.flatnonzero(mask[:, c])
    h, w = mask.shape
    return c / (w - 1), float(ys.mean()) / (h - 1)


def change_score(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute luma difference on a 4x-downsampled frame."""
    la = a[::4, ::4].mean(axis=2)
    lb = b[::4, ::4].mean(axis=2)
    return float(np.abs(la - lb).mean())


class FrameSelector:
    def __init__(self, max_fps: float, threshold: float = 1.0):
        self.max_fps = max_fps
        self.threshold = threshold
        self.last_sel_t = -1e9
        self.last_frame: np.ndarray | None = None

    def offer(self, rgb: np.ndarray, t: float) -> bool:
        if t - self.last_sel_t < 1.0 / self.max_fps:
            return False
        if self.last_frame is not None and t - self.last_sel_t < 1.0:
            if change_score(rgb, self.last_frame) < self.threshold:
                return False
        self.last_sel_t = t
        self.last_frame = rgb
        return True


class StubVisionModel:
    """Fixed random CNN (cost proxy) + chart reader; `service_ms` pads compute to a target service time."""

    def __init__(self, service_ms: float, seed: int = 0, threads: int = 2):
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 8),
        ).eval()
        self.service_s = service_ms / 1000
        self.threads = threads

    def __call__(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        torch = self.torch
        torch.set_num_threads(self.threads)
        with torch.no_grad():
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
            emb = self.net(x)
        cx, cy = chart_point(rgb)
        spent = time.perf_counter() - t0
        if spent < self.service_s:
            time.sleep(self.service_s - spent)
        return {"chart_x": cx, "chart_y": cy, "emb0": float(emb[0, 0]), "compute_s": spent}


def intent_of(text: str) -> str:
    t = text.lower()
    if "balance" in t:
        return "balance"
    if "spen" in t or "spent" in t:
        return "spending"
    if "largest" in t or "charge" in t or "flag" in t:
        return "largest"
    return "unknown"


class WhisperStt:
    name = "whisper-tiny"

    def __init__(self, threads: int = 2):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        torch.set_num_threads(threads)
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
        self.torch = torch

    def __call__(self, pcm16k: np.ndarray, oracle: str | None = None) -> str:
        feats = self.proc(pcm16k, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.no_grad():
            ids = self.model.generate(feats, max_new_tokens=64, language="en", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


class SimulatedStt:
    """Clearly-labelled fallback: returns the oracle transcript after rtf * duration seconds."""

    name = "simulated"

    def __init__(self, rtf: float = 0.1):
        self.rtf = rtf

    def __call__(self, pcm16k: np.ndarray, oracle: str | None = None) -> str:
        time.sleep(self.rtf * len(pcm16k) / 16000)
        return oracle or ""


@dataclass(order=True)
class Job:
    prio: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False)
    t_submit: float = field(compare=False, default=0.0)


class Accelerator:
    """Single worker thread shared by vision and STT jobs, under one of the policies in the module doc."""

    def __init__(self, vision: StubVisionModel, stt, policy: str, on_result, fifo_cap: int = 8):
        self.vision, self.stt, self.policy, self.on_result = vision, stt, policy, on_result
        self.fifo_cap = fifo_cap
        self.cv = threading.Condition()
        self.q: collections.deque[Job] = collections.deque()
        self.hi: collections.deque[Job] = collections.deque()  # STT/answer jobs under "latest"
        self.seq = 0
        self.stopped = False
        self.dropped_frames = 0
        self.blocked_s = 0.0
        self.busy_s = 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def submit_frame(self, payload: dict) -> bool:
        """Returns False if the frame was dropped by backpressure. Blocks only under policy 'blocking'."""
        with self.cv:
            self.seq += 1
            job = Job(1, self.seq, "vision", payload, time.monotonic())
            if self.policy == "latest":
                kept = [j for j in self.q if j.kind != "vision"]
                self.dropped_frames += len(self.q) - len(kept)
                self.q = collections.deque(kept)
                self.q.append(job)
            elif self.policy == "fifo":
                if sum(j.kind == "vision" for j in self.q) >= self.fifo_cap:
                    self.dropped_frames += 1
                    return False
                self.q.append(job)
            elif self.policy == "blocking":
                t0 = time.monotonic()
                while sum(j.kind == "vision" for j in self.q) >= 2 and not self.stopped:
                    self.cv.wait(0.05)
                self.blocked_s += time.monotonic() - t0
                self.q.append(job)
            else:  # unbounded
                self.q.append(job)
            self.cv.notify_all()
            return True

    def submit_stt(self, payload: dict) -> None:
        with self.cv:
            self.seq += 1
            job = Job(0, self.seq, "stt", payload, time.monotonic())
            (self.hi if self.policy == "latest" else self.q).append(job)
            self.cv.notify_all()

    def queue_depth(self) -> int:
        with self.cv:
            return len(self.q) + len(self.hi)

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.q and not self.hi and not self.stopped:
                    self.cv.wait(0.05)
                if self.stopped:
                    return
                job = self.hi.popleft() if self.hi else self.q.popleft()
                self.cv.notify_all()
            t0 = time.monotonic()
            if job.kind == "vision":
                res = self.vision(job.payload["rgb"])
            else:
                text = self.stt(job.payload["pcm16k"], job.payload.get("oracle"))
                res = {"text": text, "intent": intent_of(text)}
            t1 = time.monotonic()
            self.busy_s += t1 - t0
            self.on_result(job, res, t0, t1)

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()
        self.th.join(timeout=5)


def normalise_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
