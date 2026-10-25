"""WebRTC ingest leg (aiortc): client → server over real ICE/DTLS-SRTP on localhost, then AV1 transcode.

aiortc negotiates VP8/H.264 only (no AV1 RTP support), so the ingest leg carries VP8. The server decodes
and re-encodes to AV1 (libsvtav1), which feeds the AV1-over-RTP distribution/inference leg in
`media/session.py`. Glass-to-server latency is measured from a frame index stamped into the pixels as a
block barcode, so no clock or RTP-timestamp mapping across the aiortc stack is needed.
"""

from __future__ import annotations

import asyncio
import fractions
import time

import numpy as np

BITS = 16  # 12-bit frame index + 4-bit check
BLOCK = 8


def _code(idx: int) -> int:
    return (idx & 0xFFF) | ((((idx * 7) + 5) & 0xF) << 12)


def stamp(rgb: np.ndarray, idx: int) -> np.ndarray:
    out = rgb.copy()
    c = _code(idx)
    for b in range(BITS):
        out[:BLOCK, b * BLOCK : (b + 1) * BLOCK] = 255 if (c >> b) & 1 else 0
    return out


def read_stamp(rgb: np.ndarray) -> int | None:
    """Frame index, or None if the check bits do not match (corrupted stamp)."""
    c = 0
    for b in range(BITS):
        if rgb[2 : BLOCK - 2, b * BLOCK + 2 : (b + 1) * BLOCK - 2].mean() > 127:
            c |= 1 << b
    idx = c & 0xFFF
    return idx if _code(idx) == c else None


async def run_ingest(frames: np.ndarray, transcode_kbps: int = 400, threads: int = 2, on_frame=None) -> dict:
    """on_frame(idx, rgb, capture_monotonic_s), if given, is called for every received frame whose stamp
    decodes. It must not block (the M10 assistant demo hands frames to a backpressured inference worker)."""
    import av
    from aiortc import RTCConfiguration, RTCPeerConnection, VideoStreamTrack

    from media.codec import Av1Encoder, EncoderSettings

    cap_times: dict[int, float] = {}

    class SynthTrack(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self.k = 0
            self.t0 = None

        async def recv(self):
            if self.t0 is None:
                self.t0 = time.monotonic()
            k = self.k
            target = self.t0 + k / 30
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            f = av.VideoFrame.from_ndarray(stamp(frames[k % len(frames)], k), format="rgb24")
            f.pts, f.time_base = k * 3000, fractions.Fraction(1, 90000)
            cap_times[k] = time.monotonic()
            self.k += 1
            return f

    # host candidates only: with aiortc's default public STUN server the selected pair on this host reached
    # "connected" but delivered no media (observed: 0 of ~320 frames), so STUN is disabled for the loopback
    cfg = RTCConfiguration(iceServers=[])
    pc_c, pc_s = RTCPeerConnection(cfg), RTCPeerConnection(cfg)
    recv_log: list[tuple[int, float, float]] = []  # idx, latency, transcode_s
    done = asyncio.Event()
    h, w = frames.shape[1:3]
    enc = Av1Encoder(w, h, EncoderSettings(transcode_kbps, threads=threads))
    av1_bytes = 0
    codec_name = {}

    @pc_s.on("track")
    def on_track(track):
        async def consume():
            nonlocal av1_bytes
            while not done.is_set():
                try:
                    fr = await asyncio.wait_for(track.recv(), 2.0)
                except Exception:
                    break
                t_rx = time.monotonic()
                rgb = fr.to_ndarray(format="rgb24")
                idx = read_stamp(rgb)
                t0 = time.perf_counter()
                for tu, _ in enc.encode(rgb, (idx or 0) * 3000):
                    av1_bytes += len(tu)
                tc = time.perf_counter() - t0
                if idx is not None and idx in cap_times:
                    recv_log.append((idx, t_rx - cap_times[idx], tc))
                    if on_frame is not None:
                        on_frame(idx, rgb, cap_times[idx])

        asyncio.ensure_future(consume())

    pc_c.addTrack(SynthTrack())
    await pc_c.setLocalDescription(await pc_c.createOffer())
    await pc_s.setRemoteDescription(pc_c.localDescription)
    await pc_s.setLocalDescription(await pc_s.createAnswer())
    await pc_c.setRemoteDescription(pc_s.localDescription)
    for line in pc_s.localDescription.sdp.splitlines():
        if line.startswith("a=rtpmap:") and "rtx" not in line:
            codec_name.setdefault("first", line.split()[1])
    dur = len(frames) / 30
    await asyncio.sleep(dur + 1.0)
    done.set()
    await pc_c.close()
    await pc_s.close()
    lat = np.array([r[1] for r in recv_log]) if recv_log else np.array([np.nan])
    valid = len({r[0] for r in recv_log})
    return {
        "negotiated_codec": codec_name.get("first"),
        "frames_sent": len(cap_times),
        "frames_received": valid,
        "glass_to_server_ms_mean": float(np.nanmean(lat) * 1000),
        "glass_to_server_ms_p95": float(np.nanpercentile(lat, 95) * 1000),
        "av1_transcode_ms_mean": float(np.mean([r[2] for r in recv_log]) * 1000) if recv_log else None,
        "av1_kbps": av1_bytes * 8 / dur / 1000,
    }
