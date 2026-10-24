"""Offline calibration of Q(bitrate, fps) -> VMAF for the joint media controller (no network).

For each (rung, fps) the source is encoded with libsvtav1 (CBR, low-delay), decoded, and rendered at 30 fps
with frame repetition. VMAF is measured against the 30 fps source.

  uv run python -m bench.media_calibrate --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SVT_LOG", "1")

import numpy as np  # noqa: E402

from bench.meta import write_result  # noqa: E402
from media.adapt import FPS_CHOICES, LADDER_KBPS  # noqa: E402
from media.codec import Av1Decoder, Av1Encoder, EncoderSettings  # noqa: E402
from media.source import build_source  # noqa: E402
from media.vmaf import vmaf_pairs  # noqa: E402


def encode_render(frames: np.ndarray, kbps: int, fps: int, threads: int) -> tuple[list[np.ndarray], float]:
    enc = Av1Encoder(frames.shape[2], frames.shape[1], EncoderSettings(kbps, fps, threads=threads))
    dec = Av1Decoder()
    step = 30 // fps
    out, last, nbytes = [], None, 0
    for k, f in enumerate(frames):
        if k % step == 0:
            for tu, _ in enc.encode(f, k * 3000):
                nbytes += len(tu)
                d = dec.decode(tu)
                if d:
                    last = d[-1]
        out.append(last if last is not None else np.zeros_like(f))
    return out, nbytes * 8 / (len(frames) / 30) / 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        src = build_source(seed, duration_s=a.duration, use_say=False)
        ref = list(src.frames[30:])  # skip 1 s start-up
        for kbps in LADDER_KBPS:
            for fps in FPS_CHOICES:
                dist, actual = encode_render(src.frames, kbps, fps, a.threads)
                v = vmaf_pairs(ref, dist[30:], threads=a.threads)
                rows.append(
                    {"seed": seed, "kbps": kbps, "fps": fps, "vmaf": v["mean"], "actual_kbps": actual}
                )
                print(seed, kbps, fps, round(v["mean"], 2), round(actual, 1))
    table = {}
    for kbps in LADDER_KBPS:
        for fps in FPS_CHOICES:
            vs = [r["vmaf"] for r in rows if r["kbps"] == kbps and r["fps"] == fps]
            table[f"{kbps}@{fps}"] = float(np.mean(vs))
    write_result("m6/calibration.json", {"rows": rows, "table": table}, config=vars(a))


if __name__ == "__main__":
    main()
