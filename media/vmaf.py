"""VMAF of the *rendered* stream (frame repetition on freezes / reduced fps) against the source, via ffmpeg libvmaf.

Caveat: the default VMAF model (vmaf_v0.6.1) was trained on 1080p TV viewing; frames here are 256x256
synthetic screen content, so absolute scores are not comparable to published VMAF numbers. Only
relative comparisons between conditions are meaningful.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def ffmpeg_bin() -> str | None:
    for c in ("/opt/homebrew/bin/ffmpeg", shutil.which("ffmpeg")):
        if c and Path(c).exists():
            return c
    return None


def vmaf_pairs(ref: list[np.ndarray], dist: list[np.ndarray], threads: int = 4) -> dict:
    ff = ffmpeg_bin()
    if ff is None or not ref:
        return {"mean": None, "p5": None, "n": 0, "per_frame": []}
    h, w = ref[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        rp, dp, lp = Path(td) / "ref.rgb", Path(td) / "dist.rgb", Path(td) / "vmaf.json"
        rp.write_bytes(b"".join(np.ascontiguousarray(f).tobytes() for f in ref))
        dp.write_bytes(b"".join(np.ascontiguousarray(f).tobytes() for f in dist))
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30"]
        cmd = [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            *raw,
            "-i",
            str(dp),
            *raw,
            "-i",
            str(rp),
            "-lavfi",
            f"[0:v]format=yuv420p[d];[1:v]format=yuv420p[r];[d][r]libvmaf=log_fmt=json:log_path={lp}:n_threads={threads}",
            "-f",
            "null",
            "-",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        d = json.loads(lp.read_text())
    per = [fr["metrics"]["vmaf"] for fr in d["frames"]]
    return {"mean": float(np.mean(per)), "p5": float(np.percentile(per, 5)), "n": len(per), "per_frame": per}


def rendered_vmaf(
    src_frames: np.ndarray,
    display: list[tuple[float, int, int]],
    decoded: dict[int, np.ndarray],
    t_lo: float,
    t_hi: float,
) -> dict:
    black = np.zeros_like(src_frames[0])
    ref, dist = [], []
    for t, r, s in display:
        if t_lo <= t <= t_hi and 0 <= r < len(src_frames):
            ref.append(src_frames[r])
            dist.append(decoded.get(s, black) if s >= 0 else black)
    return vmaf_pairs(ref, dist)
