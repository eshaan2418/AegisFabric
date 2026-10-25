"""One media trial: Sender → emulated impaired link → Receiver, with RTCP feedback on the reverse link.

All components are asyncio tasks on ONE event loop, and they exchange byte-exact RTP/RTCP packets over an
in-process emulated link (`media.netem._Dir`: delay, jitter, Gilbert-Elliott loss, rate limit, drop-tail).

Two execution modes (`Runtime`):
* ``virtual`` (used for all reported M6 numbers): a virtual-time loop (`media.vloop`). Codec, STT and
  vision calls really execute, single-threaded, and each call's measured CPU time (`time.process_time`) is
  charged as its latency in virtual time. Results do not depend on host load, which on the shared dev machine
  reached load-average >200 on 18 cores and made wall-clock real-time runs meaningless.
* ``realtime``: a standard asyncio loop; codec/model calls run in a thread pool, wall-clock timing (the
  vision stub pads itself to its service time). Used for the live demo/test only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field

import numpy as np

os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1: errors only

from media import rtp as R  # noqa: E402
from media.adapt import Decision, DelayBasedEstimator, IndependentPolicy, JointPolicy, LossBasedRate  # noqa: E402
from media.av1rtp import Depacketizer, packetize  # noqa: E402
from media.codec import Av1Decoder, Av1Encoder, EncoderSettings, OpusDecoder, OpusEncoder  # noqa: E402
from media.inference import Accelerator, FrameSelector, chart_point  # noqa: E402
from media.netem import REVERSE, LinkProfile, _Dir  # noqa: E402
from media.source import FPS, SR, Source  # noqa: E402
from media.vloop import VirtualTimeLoop  # noqa: E402

VIDEO_PT, AUDIO_PT = 96, 111
LATE_GRACE_S = 0.25
VSSRC, ASSRC, RSSRC = 0x1111, 0x2222, 0x3333
NTP_OFF = 1.7e9  # loop time -> synthetic NTP wall time (both ends share it; only differences matter)


def loop_to_ntp(t: float) -> tuple[int, int]:
    return R.ntp_now(t + NTP_OFF)


def ntp_to_loop(msw: int, lsw: int) -> float:
    return R.ntp_to_unix(msw, lsw) - NTP_OFF


class Runtime:
    def __init__(self, mode: str):
        assert mode in ("virtual", "realtime")
        self.mode = mode
        self.pool = concurrent.futures.ThreadPoolExecutor(4) if mode == "realtime" else None
        self.cpu: dict[str, list[float]] = {}

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def compute(self, fn, *args, min_s: float = 0.0, remote: bool = False, tag: str | None = None):
        """Run `fn(*args)` and account for its latency. virtual: max(CPU time, min_s) is charged to virtual
        time. realtime: wall time passes naturally (work runs in a thread pool)."""
        if self.mode == "virtual":
            c0 = time.process_time()
            r = fn(*args)
            cost = time.process_time() - c0
            self.cpu.setdefault(tag or getattr(fn, "__name__", "fn"), []).append(cost)
            await asyncio.sleep(max(cost, min_s))
            return r
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, fn, *args)

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown(wait=False)


class Link:
    """Byte-exact datagram delivery through an impaired direction, scheduled on the loop."""

    def __init__(self, prof: LinkProfile, seed: int, t0: float, deliver):
        self.d = _Dir(prof, np.random.default_rng(seed), t0)
        self.deliver = deliver

    def send(self, data: bytes) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        at = self.d.schedule(len(data), now)
        if at is not None:
            loop.call_at(at, self.deliver, data)


@dataclass
class TrialConfig:
    profile: LinkProfile
    policy: str  # "joint" | "independent" | "blocking"
    seed: int
    vision_ms: float = 50.0
    start_kbps: int = 400
    threads: int = 1


@dataclass
class TrialLog:
    sent_frames: dict[int, float] = field(default_factory=dict)  # idx -> capture time
    sent_bytes: int = 0
    sent_audio_bytes: int = 0
    decisions: list[tuple] = field(default_factory=list)
    reinits: int = 0
    keyframes: int = 0
    plis: int = 0
    rtts: list[float] = field(default_factory=list)
    decoded: dict[int, np.ndarray] = field(default_factory=dict)
    decoded_t: dict[int, float] = field(default_factory=dict)
    lost_frames: int = 0
    late_frames: int = 0
    retransmits: int = 0
    nacked: int = 0
    send_lag: list[float] = field(default_factory=list)
    undecodable: int = 0
    recv_bytes: int = 0
    display: list[tuple[float, int, int]] = field(default_factory=list)  # (t, ref idx, shown idx | -1)
    vision: list[dict] = field(default_factory=list)
    stt: list[dict] = field(default_factory=list)
    av_skew_ms: list[float] = field(default_factory=list)
    utterances: list[tuple[float, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class Sender:
    def __init__(self, rt: Runtime, src: Source, cfg: TrialConfig, log: TrialLog, policy, t0: float):
        self.rt, self.src, self.cfg, self.log, self.policy = rt, src, cfg, log, policy
        self.t0 = t0
        h, w = src.frames.shape[1:3]
        self.dec = Decision(cfg.start_kbps, 30, 10.0, "fifo")
        if isinstance(policy, JointPolicy):
            self.dec = Decision(cfg.start_kbps, policy.best_fps(cfg.start_kbps), 10.0, "latest")
        self.enc = Av1Encoder(w, h, EncoderSettings(self.dec.bitrate_kbps, self.dec.fps, threads=cfg.threads))
        self.opus = OpusEncoder()
        self.loss = LossBasedRate(cfg.start_kbps)
        self.remb = float("inf")
        self.service_fps = 1000.0 / cfg.vision_ms
        self.last_change = t0
        self.vseq = random.Random(cfg.seed).randrange(1 << 16)
        self.aseq = 0
        self.vcount = self.vbytes = self.acount = self.abytes = 0
        self.last_pli_kf = -1e9
        self.history: dict[int, bytes] = {}
        self.link: Link | None = None
        self.pending_cfg: EncoderSettings | None = None

    def send(self, data: bytes) -> None:
        self.link.send(data)

    async def run(self) -> None:
        dur = len(self.src.frames) / FPS
        await asyncio.gather(self._video(), self._audio(), self._rtcp(dur), self._adapt(dur))

    async def _sleep_until(self, t: float) -> None:
        d = t - self.rt.now()
        if d > 0:
            await asyncio.sleep(d)

    async def _video(self) -> None:
        for k in range(len(self.src.frames)):
            cap = self.t0 + k / FPS
            await self._sleep_until(cap)
            if self.pending_cfg is not None:  # encoder re-init (cost charged like any compute)
                cfg, self.pending_cfg = self.pending_cfg, None
                await self.rt.compute(self.enc.reconfigure, cfg, tag="enc_reinit")
            if k % (30 // self.dec.fps):
                continue
            self.log.send_lag.append(self.rt.now() - cap)
            out = await self.rt.compute(self.enc.encode, self.src.frames[k], k * 3000, tag="av1_encode")
            for tu, kf in out:
                self.log.keyframes += kf
                self.log.sent_frames[k] = cap
                self.log.sent_bytes += len(tu)
                for payload, marker in packetize(tu):
                    ast = R.abs_send_time(self.rt.now())
                    pkt = R.RtpPacket(VIDEO_PT, self.vseq, k * 3000, VSSRC, payload, marker, ast).pack()
                    self.history[self.vseq] = pkt
                    self.history.pop((self.vseq - 2000) & 0xFFFF, None)
                    self.vseq = (self.vseq + 1) & 0xFFFF
                    self.vcount += 1
                    self.vbytes += len(payload)
                    self.send(pkt)

    async def _audio(self) -> None:
        a = self.src.audio
        for i in range(len(a) // 960):
            await self._sleep_until(self.t0 + i * 0.02)
            for p in self.opus.encode(a[i * 960 : (i + 1) * 960]):  # ~0.1 ms, not charged
                self.send(R.RtpPacket(AUDIO_PT, self.aseq, i * 960, ASSRC, p).pack())
                self.aseq = (self.aseq + 1) & 0xFFFF
                self.acount += 1
                self.abytes += len(p)
                self.log.sent_audio_bytes += len(p)

    async def _rtcp(self, dur: float) -> None:
        while self.rt.now() < self.t0 + dur:
            now = self.rt.now()
            msw, lsw = loop_to_ntp(now)
            vts, ats = int((now - self.t0) * 90000), int((now - self.t0) * SR)
            self.send(R.pack_sr(R.SenderReport(VSSRC, msw, lsw, vts, self.vcount, self.vbytes)))
            self.send(R.pack_sr(R.SenderReport(ASSRC, msw, lsw, ats, self.acount, self.abytes)))
            await asyncio.sleep(0.25)

    def on_rtcp(self, data: bytes) -> None:
        now = self.rt.now()
        for p in R.unpack_rtcp(data):
            if isinstance(p, R.ReceiverReport):
                for b in p.blocks:
                    if b.ssrc == VSSRC:
                        self.loss.on_loss(b.fraction_lost / 256)
                        rtt = R.rtt_from_block(b, now + NTP_OFF)
                        if rtt is not None:
                            self.log.rtts.append(rtt)
            elif isinstance(p, R.Remb):
                self.remb = p.bitrate_bps / 1000
            elif isinstance(p, R.Pli):
                if now - self.last_pli_kf > 0.3:
                    self.enc.request_keyframe()
                    self.last_pli_kf = now
                    self.log.plis += 1
            elif isinstance(p, R.Nack):
                for q in p.seqs:
                    h = self.history.get(q)
                    if h is not None:
                        self.send(h)
                        self.log.retransmits += 1
            elif isinstance(p, R.AppFeedback):
                self.service_fps = p.service_fps_milli / 1000

    async def _adapt(self, dur: float) -> None:
        while self.rt.now() < self.t0 + dur:
            await asyncio.sleep(0.2)
            now = self.rt.now()
            target = min(self.loss.rate, self.remb)
            new = self.policy.decide(target, self.dec, now - self.last_change, self.service_fps)
            if (new.bitrate_kbps, new.fps) != (self.dec.bitrate_kbps, self.dec.fps):
                self.pending_cfg = EncoderSettings(new.bitrate_kbps, new.fps, threads=self.cfg.threads)
                self.log.reinits += 1
                self.last_change = now
            self.dec = new
            self.log.decisions.append(
                (now - self.t0, new.bitrate_kbps, new.fps, new.select_fps, self.loss.rate, self.remb)
            )


class Receiver:
    def __init__(self, rt: Runtime, src: Source, cfg: TrialConfig, log: TrialLog, t0: float, vision, stt):
        self.rt, self.src, self.cfg, self.log, self.t0 = rt, src, cfg, log, t0
        self.vstats = R.ReceiverStats(VSSRC, 90000)
        self.astats = R.ReceiverStats(ASSRC, SR)
        self.bwe = DelayBasedEstimator()
        self.sr_map: dict[int, tuple[float, int, int]] = {}  # ssrc -> (capture time, rtp ts, clock)
        self.frames: dict[int, dict] = {}  # ts -> {"pkts": {seq: (payload, marker)}, "cap"}
        self.last_marker_seq: int | None = None
        self.last_played_ts = -1
        self.broken = False
        self.dec = Av1Decoder()
        self.opus = OpusDecoder()
        self.jitter_delay = 0.08
        self.select_fps = 10.0
        qpol = {"independent": "fifo", "blocking": "blocking"}.get(cfg.policy, "latest")
        self.acc = Accelerator(rt, vision, cfg.vision_ms / 1000, stt, qpol, self.on_result)
        self.selector = FrameSelector(self.select_fps)
        self.svc_times: list[float] = []
        self.last_pli = -1e9
        self.audio_buf: list[np.ndarray] = []
        self.in_utt = False
        self.silence_run = 0
        self.utt_start_cap = 0.0
        self.latest_vision: dict | None = None
        self.min_cap_owd = float("inf")
        self.missing: dict[int, list] = {}
        self.max_vseq: int | None = None
        self.last_audio: tuple[float, float] | None = None
        self.link: Link | None = None

    def capture_time(self, ssrc: int, ts: int) -> float:
        m = self.sr_map.get(ssrc)
        if m is None:  # before the first SR: nominal start
            return self.t0 + ts / (90000 if ssrc == VSSRC else SR)
        cap, rts, clock = m
        d = ((ts - rts + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
        return cap + d / clock

    def on_result(self, job, res, t0, t1) -> None:
        if job.kind == "vision":
            self.svc_times.append(t1 - t0)
            idx = job.payload["idx"]
            ref = chart_point(self.src.frames[idx])
            rec = {
                "idx": idx,
                "t_done": t1,
                "glass_to_inf": t1 - job.payload["cap"],
                "queue_s": t0 - job.t_submit,
                "chart_err": abs(res["chart_y"] - ref[1]) + abs(res["chart_x"] - ref[0]),
                "y": res["chart_y"],
                "x": res["chart_x"],
            }
            self.log.vision.append(rec)
            self.latest_vision = rec
        else:
            lv = self.latest_vision
            self.log.stt.append(
                {
                    "t_done": t1,
                    "text": res["text"],
                    "intent": res["intent"],
                    "utt_end_cap": job.payload["end_cap"],
                    "queue_s": t0 - job.t_submit,
                    "service_s": t1 - t0,
                    "vision_idx": lv["idx"] if lv else None,
                    "vision_xy": (lv["x"], lv["y"]) if lv else None,
                }
            )

    async def run(self) -> None:
        end = self.t0 + len(self.src.frames) / FPS + 1.0
        acc_task = asyncio.ensure_future(self.acc.run())
        await asyncio.gather(
            self._playout(end), self._feedback(end), self._display(end), self._nack_loop(end)
        )
        # drain outstanding STT/vision work (bounded)
        deadline = self.rt.now() + 15
        while self.acc.queue_depth() and self.rt.now() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0)
        self.acc.stop()
        try:
            await asyncio.wait_for(acc_task, 30)
        except (TimeoutError, asyncio.CancelledError):
            acc_task.cancel()

    def on_datagram(self, data: bytes) -> None:
        now = self.rt.now()
        if R.is_rtcp(data):
            for p in R.unpack_rtcp(data):
                if isinstance(p, R.SenderReport):
                    clock = 90000 if p.ssrc == VSSRC else SR
                    self.sr_map[p.ssrc] = (ntp_to_loop(p.ntp_msw, p.ntp_lsw), p.rtp_ts, clock)
                    (self.vstats if p.ssrc == VSSRC else self.astats).on_sr(p, now)
            return
        pkt = R.RtpPacket.unpack(data)
        if pkt.pt == VIDEO_PT:
            self.log.recv_bytes += len(pkt.payload)
            self.vstats.on_packet(pkt.seq, pkt.ts, now)
            cap = self.capture_time(VSSRC, pkt.ts)
            self._track_seq(pkt.seq, now)
            first_of_frame = pkt.ts not in self.frames
            self.min_cap_owd = min(self.min_cap_owd, now - cap)
            send_owd = R.ast_delta(now, pkt.abs_send_time) if pkt.abs_send_time is not None else now - cap
            self.bwe.on_packet(now, len(data), send_owd if first_of_frame else None)
            if pkt.ts in self.frames or self.last_played_ts < pkt.ts:
                f = self.frames.setdefault(pkt.ts, {"pkts": {}, "cap": cap})
                f["pkts"][pkt.seq] = (pkt.payload, pkt.marker)
        elif pkt.pt == AUDIO_PT:
            self.astats.on_packet(pkt.seq, pkt.ts, now)
            self._on_audio(pkt, now)

    def _track_seq(self, seq: int, now: float) -> None:
        self.missing.pop(seq, None)
        if self.max_vseq is None:
            self.max_vseq = seq
            return
        d = (seq - self.max_vseq) & 0xFFFF
        if 0 < d < 0x8000:
            for k in range(1, min(d, 200)):
                self.missing[(self.max_vseq + k) & 0xFFFF] = [now, 0, -1e9]  # first-seen, tries, last-nack
            self.max_vseq = seq

    async def _nack_loop(self, end: float) -> None:
        while self.rt.now() < end:
            await asyncio.sleep(0.01)
            now = self.rt.now()
            due = []
            for q, st in list(self.missing.items()):
                if now - st[0] > 0.5 or st[1] >= 3:
                    del self.missing[q]
                elif now - st[2] > 0.08:
                    st[1] += 1
                    st[2] = now
                    due.append(q)
            if due:
                self.log.nacked += len(due)
                self.link.send(R.pack_nack(R.Nack(RSSRC, VSSRC, due)))

    def _on_audio(self, pkt: R.RtpPacket, now: float) -> None:
        pcm = self.opus.decode(pkt.payload)  # ~0.1 ms, not charged
        cap = self.capture_time(ASSRC, pkt.ts)
        self.last_audio = (max(now, cap + self.jitter_delay), cap)  # common A/V playout offset via SR mapping
        e = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0  # energy VAD
        if e > 400:
            if not self.in_utt:
                self.in_utt, self.utt_start_cap, self.audio_buf = True, cap, []
            self.silence_run = 0
        elif self.in_utt:
            self.silence_run += 1
        if self.in_utt:
            self.audio_buf.append(pcm)
            if self.silence_run >= 20:  # 400 ms of silence ends the utterance
                self.in_utt = False
                pcm48 = np.concatenate(self.audio_buf)
                pcm16 = pcm48[: len(pcm48) // 3 * 3].reshape(-1, 3).mean(axis=1) / 32768.0
                end_cap = cap - 0.4
                self.log.utterances.append((self.utt_start_cap, end_cap))
                rel = self.utt_start_cap - self.t0
                qs = sorted(self.src.questions, key=lambda q: abs(q.start_s - rel))
                self.acc.submit_stt(
                    {
                        "pcm16k": pcm16.astype(np.float32),
                        "end_cap": end_cap,
                        "oracle": qs[0].text if qs else "",
                    }
                )

    async def _playout(self, end: float) -> None:
        while self.rt.now() < end:
            await asyncio.sleep(0.004)
            for ts in sorted(self.frames):
                now = self.rt.now()
                f = self.frames[ts]
                due = f["cap"] + self.jitter_delay
                if due > now:
                    break
                # a late frame is waited for (it is a decoding reference) up to a grace period
                if not self._frame_info(f)[3] and now < due + LATE_GRACE_S and len(self.frames) < 15:
                    break
                if now - due > 0.005:
                    self.log.late_frames += 1
                    self.jitter_delay = min(0.4, self.jitter_delay + 0.5 * (now - due))
                del self.frames[ts]
                self.last_played_ts = ts
                await self._play(ts, f)

    def _frame_info(self, f: dict) -> tuple:
        pk = f["pkts"]
        lo = min(pk)
        seqs = sorted(pk, key=lambda s: (s - lo) & 0xFFFF)
        first, last = seqs[0], seqs[-1]
        contiguous = ((last - first) & 0xFFFF) + 1 == len(seqs)
        has_marker = pk[last][1]
        first_pl = pk[first][0]
        starts_cvs = bool(first_pl[0] & 0x08) and not (first_pl[0] & 0x80)
        prev_ok = self.last_marker_seq is not None and ((first - self.last_marker_seq) & 0xFFFF) == 1
        return (
            seqs,
            first,
            last,
            contiguous and has_marker and (prev_ok or starts_cvs),
            has_marker,
            starts_cvs,
        )

    async def _play(self, ts: int, f: dict) -> None:
        seqs, first, last, complete, has_marker, starts_cvs = self._frame_info(f)
        idx = ts // 3000
        if has_marker:
            self.last_marker_seq = last
        if not complete:
            self.log.lost_frames += 1
            self.broken = True
            self._pli()
            return
        if self.broken and not starts_cvs:
            self.log.undecodable += 1
            self._pli()
            return
        try:
            tu = Depacketizer.assemble([f["pkts"][s][0] for s in seqs])
        except ValueError:
            self.log.lost_frames += 1
            self.broken = True
            return
        frames = await self.rt.compute(self.dec.decode, tu, tag="av1_decode")
        if not frames:
            return
        self.broken = False
        now = self.rt.now()
        rgb = frames[-1]
        self.log.decoded[idx] = rgb
        self.log.decoded_t[idx] = now
        if self.last_audio is not None:
            a_render, a_cap = self.last_audio
            self.log.av_skew_ms.append(1000 * ((now - f["cap"]) - (a_render - a_cap)))
        self.selector.max_fps = self.select_fps
        if self.selector.offer(rgb, now):
            await self.acc.submit_frame({"rgb": rgb, "idx": idx, "cap": f["cap"]})

    def _pli(self) -> None:
        now = self.rt.now()
        if now - self.last_pli > 0.25:
            self.link.send(R.pack_pli(R.Pli(RSSRC, VSSRC)))
            self.last_pli = now

    async def _feedback(self, end: float) -> None:
        while self.rt.now() < end:
            await asyncio.sleep(0.2)
            now = self.rt.now()
            # adaptive playout delay (from capture): base one-way delay + 3x RFC3550 jitter + 20 ms, bounded
            jit_s = self.vstats.jitter / 90000
            base = self.min_cap_owd if np.isfinite(self.min_cap_owd) else 0.05
            target = min(max(base + 3 * jit_s + 0.02, 0.04), 0.4)
            self.jitter_delay += (target - self.jitter_delay) * (0.5 if target > self.jitter_delay else 0.1)
            svc = statistics.mean(self.svc_times[-10:]) if self.svc_times else self.cfg.vision_ms / 1000
            service_fps = 1.0 / max(svc, 1e-3)
            if self.cfg.policy == "joint":
                self.select_fps = max(1.0, min(10.0, 0.8 * service_fps))
            pkt = R.pack_rr(
                R.ReceiverReport(RSSRC, [self.vstats.report_block(now), self.astats.report_block(now)])
            )
            pkt += R.pack_remb(R.Remb(RSSRC, int(self.bwe.update(now)), [VSSRC]))
            pkt += R.pack_app(R.AppFeedback(RSSRC, int(service_fps * 1000), self.acc.queue_depth()))
            self.link.send(pkt)

    async def _display(self, end: float) -> None:
        k, shown = 0, -1
        while True:
            t = self.t0 + k / FPS + self.jitter_delay
            if t > end:
                break
            await asyncio.sleep(max(0.0, t - self.rt.now()))
            now = self.rt.now()
            cand = [i for i, td in self.log.decoded_t.items() if td <= now and i > shown]
            if cand:
                shown = max(cand)
            ref = int(round((now - self.jitter_delay - self.t0) * FPS))
            self.log.display.append((now - self.t0, ref, shown))
            k += 1


async def _trial(rt: Runtime, src: Source, cfg: TrialConfig, vision, stt, quality_table: dict) -> TrialLog:
    log = TrialLog()
    policy = JointPolicy(quality_table) if cfg.policy == "joint" else IndependentPolicy()
    t0 = rt.now() + 0.5
    snd = Sender(rt, src, cfg, log, policy, t0)
    rcv = Receiver(rt, src, cfg, log, t0, vision, stt)
    ss = np.random.SeedSequence(cfg.seed).spawn(2)
    snd.link = Link(cfg.profile, ss[0], t0, rcv.on_datagram)
    rcv.link = Link(REVERSE, ss[1], t0, snd.on_rtcp)
    await asyncio.gather(snd.run(), rcv.run())
    log.extra.update(
        t_start=t0,
        relay={"fwd": snd.link.d.stats, "rev": rcv.link.d.stats},
        jitter_delay_final=rcv.jitter_delay,
        dropped_backpressure=rcv.acc.dropped_frames,
        blocked_s=rcv.acc.blocked_s,
        owd_trace=rcv.bwe.trace,
        cpu_ms={k: float(np.mean(v) * 1000) for k, v in rt.cpu.items()},
        cpu_ms_p95={k: float(np.percentile(v, 95) * 1000) for k, v in rt.cpu.items()},
    )
    return log


def run_trial(
    src: Source, cfg: TrialConfig, vision, stt, quality_table: dict, mode: str = "virtual"
) -> TrialLog:
    """`vision(rgb)->dict` and `stt(pcm16k, oracle)->str` callables. In virtual mode they must run in-process
    single-threaded (e.g. `StubVisionModel(pad=False, threads=1)`, `WhisperStt(threads=1)`)."""
    rt = Runtime(mode)
    try:
        if mode == "virtual":
            loop = VirtualTimeLoop()
            try:
                return loop.run_until_complete(_trial(rt, src, cfg, vision, stt, quality_table))
            finally:
                loop.close()
        return asyncio.run(_trial(rt, src, cfg, vision, stt, quality_table))
    finally:
        rt.close()


def summarize_trial(src: Source, cfg: TrialConfig, log: TrialLog, vmaf: dict | None) -> dict:
    dur = len(src.frames) / FPS
    disp = [(t, r, s) for t, r, s in log.display if 1.0 <= t <= dur and 0 <= r < len(src.frames)]
    ages = [(r - s) / FPS if s >= 0 else dur for _, r, s in disp]
    stalls, stall_time, in_stall = 0, 0.0, False
    for a in ages:
        if a > 0.2:
            stall_time += 1 / FPS
            stalls += 0 if in_stall else 1
            in_stall = True
        else:
            in_stall = False
    t_start = log.extra["t_start"]
    g2a, correct, used = [], [], set()
    for q in src.questions:
        q_end = t_start + q.end_s
        match = [i for i, s in enumerate(log.stt) if abs(s["utt_end_cap"] - q_end) < 1.5 and i not in used]
        if not match:
            g2a.append(float("nan"))
            correct.append(0)
            continue
        used.add(match[0])
        s = log.stt[match[0]]
        g2a.append(s["t_done"] - q_end)
        ok = s["intent"] == q.intent and s["vision_xy"] is not None
        if ok:
            rx, ry = chart_point(src.frames[min(len(src.frames) - 1, int(q.end_s * FPS))])
            ok = abs(s["vision_xy"][0] - rx) < 0.05 and abs(s["vision_xy"][1] - ry) < 0.05
        correct.append(int(ok))
    sent = len(log.sent_frames)
    g2i = [v["glass_to_inf"] for v in log.vision]
    x = log.extra
    finite = [v for v in g2a if np.isfinite(v)]
    return {
        "profile": cfg.profile.name,
        "policy": cfg.policy,
        "seed": cfg.seed,
        "vision_ms": cfg.vision_ms,
        "vmaf_mean": vmaf["mean"] if vmaf else None,
        "vmaf_p5": vmaf["p5"] if vmaf else None,
        "sent_kbps": log.sent_bytes * 8 / dur / 1000,
        "recv_kbps": log.recv_bytes * 8 / dur / 1000,
        "sent_frames": sent,
        "decoded_frames": len(log.decoded),
        "dropped_network": log.lost_frames + log.undecodable,
        "late_frames": log.late_frames,
        "retransmits": log.retransmits,
        "dropped_network_frac": (sent - len(log.decoded)) / max(sent, 1),
        "dropped_backpressure": x["dropped_backpressure"],
        "stalls": stalls,
        "stall_time_s": stall_time,
        "glass_to_answer_s": [float(v) for v in g2a],
        "glass_to_answer_mean_s": float(np.mean(finite)) if finite else None,
        "answered_frac": len(finite) / max(len(src.questions), 1),
        "answer_correct_frac": float(np.mean(correct)) if correct else None,
        "glass_to_inference_p50_s": float(np.percentile(g2i, 50)) if g2i else None,
        "glass_to_inference_p95_s": float(np.percentile(g2i, 95)) if g2i else None,
        "inferred_frames": len(log.vision),
        "chart_err_mean": float(np.mean([v["chart_err"] for v in log.vision])) if log.vision else None,
        "av_skew_ms_median_abs": float(np.median(np.abs(log.av_skew_ms))) if log.av_skew_ms else None,
        "rtt_ms_median": float(np.median(log.rtts) * 1000) if log.rtts else None,
        "reinits": log.reinits,
        "keyframes": log.keyframes,
        "plis": log.plis,
        "blocked_s": x["blocked_s"],
        "mean_bitrate_decision_kbps": float(np.mean([d[1] for d in log.decisions]))
        if log.decisions
        else None,
        "mean_fps_decision": float(np.mean([d[2] for d in log.decisions])) if log.decisions else None,
        "stt_texts": [s["text"] for s in log.stt],
        "stt_service_s": [s["service_s"] for s in log.stt],
        "cpu_ms": x["cpu_ms"],
        "cpu_ms_p95": x["cpu_ms_p95"],
        "relay": x["relay"],
        "profile_cfg": asdict(cfg.profile),
    }
