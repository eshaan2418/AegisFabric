"""AV1 (libsvtav1 encode, libdav1d decode) and Opus wrappers over PyAV.

SVT-AV1 cannot change target bitrate or frame rate in-flight through PyAV, so a rate/fps change
re-creates the encoder; the new encoder starts with a keyframe (sequence header). This mirrors a
non-simulcast ABR ladder and its keyframe cost is included in all measurements.
"""

from __future__ import annotations

import fractions
import os
from dataclasses import dataclass

import av
import numpy as np

os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1 logs errors only


@dataclass(frozen=True)
class EncoderSettings:
    bitrate_kbps: int
    fps: int = 30
    preset: int = 12
    threads: int = 4
    gop: int = 60


class Av1Encoder:
    def __init__(self, width: int, height: int, settings: EncoderSettings):
        self.w, self.h = width, height
        self.settings = settings
        self._open()

    def _open(self) -> None:
        s = self.settings
        ctx = av.CodecContext.create("libsvtav1", "w")
        ctx.width, ctx.height = self.w, self.h
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = fractions.Fraction(1, 90000)
        ctx.framerate = fractions.Fraction(s.fps, 1)
        ctx.bit_rate = s.bitrate_kbps * 1000
        ctx.gop_size = s.gop
        ctx.thread_count = s.threads
        # low-delay real-time: no lookahead / scene-change, CBR rate control, single-pass
        ctx.options = {
            "preset": str(s.preset),
            "svtav1-params": f"rc=2:pred-struct=1:lookahead=0:scd=0:lp={s.threads}:tune=0",
        }
        ctx.open()
        self.ctx = ctx
        self._force_kf = True

    def reconfigure(self, settings: EncoderSettings) -> None:
        if settings != self.settings:
            self.settings = settings
            self._open()

    def request_keyframe(self) -> None:
        self._force_kf = True

    def encode(self, rgb: np.ndarray, pts90k: int) -> list[tuple[bytes, bool]]:
        """Encode one RGB frame; returns [(temporal_unit_bytes, is_keyframe)] (0 or 1 items for low-delay)."""
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        frame.pts = pts90k
        if self._force_kf:
            frame.pict_type = av.video.frame.PictureType.I
            self._force_kf = False
        out = []
        for pkt in self.ctx.encode(frame):
            out.append((bytes(pkt), bool(pkt.is_keyframe)))
        return out


class Av1Decoder:
    def __init__(self) -> None:
        self.ctx = av.CodecContext.create("libdav1d", "r")
        # zero decoder delay: one TU in -> its frame out (frame threading would add latency and misalign frames)
        self.ctx.thread_count = 1
        self.ctx.options = {"max_frame_delay": "1"}

    def decode(self, tu: bytes) -> list[np.ndarray]:
        pkt = av.Packet(tu)
        try:
            frames = self.ctx.decode(pkt)
        except av.error.FFmpegError:
            return []
        return [f.to_ndarray(format="rgb24") for f in frames]


class OpusEncoder:
    """48 kHz mono Opus, 20 ms frames."""

    def __init__(self, bitrate: int = 24000):
        self.ctx = av.CodecContext.create("libopus", "w")
        self.ctx.sample_rate = 48000
        self.ctx.layout = "mono"
        self.ctx.format = "s16"
        self.ctx.bit_rate = bitrate
        self.ctx.time_base = fractions.Fraction(1, 48000)
        self.ctx.options = {"application": "voip", "frame_duration": "20"}
        self.ctx.open()
        self._pts = 0

    def encode(self, pcm960: np.ndarray) -> list[bytes]:
        f = av.AudioFrame.from_ndarray(pcm960.reshape(1, -1).astype(np.int16), format="s16", layout="mono")
        f.sample_rate = 48000
        f.pts = self._pts
        self._pts += pcm960.size
        return [bytes(p) for p in self.ctx.encode(f)]


class OpusDecoder:
    def __init__(self) -> None:
        self.ctx = av.CodecContext.create("libopus", "r")
        self.ctx.sample_rate = 48000
        self.ctx.layout = "mono"

    def decode(self, payload: bytes) -> np.ndarray:
        try:
            frames = self.ctx.decode(av.Packet(payload))
        except av.error.FFmpegError:
            return np.zeros(960, np.float32)
        if not frames:
            return np.zeros(0, np.float32)
        return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]).astype(np.float32)
