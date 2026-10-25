"""Minimal RTP (RFC 3550) and RTCP (SR, RR, PLI per RFC 4585, REMB per draft-alvestrand-rmcat-remb).

Only the fields this pipeline uses are implemented; packets are byte-compatible with the RFC formats.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

ABS_SEND_TIME_ID = 3
NTP_EPOCH_OFFSET = 2208988800  # seconds between 1900-01-01 and 1970-01-01


def ntp_now(t: float | None = None) -> tuple[int, int]:
    t = time.time() if t is None else t
    sec = int(t) + NTP_EPOCH_OFFSET
    frac = int((t % 1.0) * (1 << 32)) & 0xFFFFFFFF
    return sec & 0xFFFFFFFF, frac


def ntp_to_unix(msw: int, lsw: int) -> float:
    return msw - NTP_EPOCH_OFFSET + lsw / (1 << 32)


def ntp_mid32(msw: int, lsw: int) -> int:
    return ((msw & 0xFFFF) << 16) | (lsw >> 16)


@dataclass
class RtpPacket:
    pt: int
    seq: int
    ts: int
    ssrc: int
    payload: bytes
    marker: bool = False
    abs_send_time: int | None = None  # 24-bit 6.18 fixed-point seconds (RFC 5285 one-byte ext, id 3)

    def pack(self) -> bytes:
        b0 = 0x80 | (0x10 if self.abs_send_time is not None else 0)  # V=2, X if extension
        b1 = (0x80 if self.marker else 0) | (self.pt & 0x7F)
        hdr = struct.pack("!BBHII", b0, b1, self.seq & 0xFFFF, self.ts & 0xFFFFFFFF, self.ssrc)
        if self.abs_send_time is not None:
            ast = self.abs_send_time & 0xFFFFFF
            hdr += struct.pack("!HHB", 0xBEDE, 1, (ABS_SEND_TIME_ID << 4) | 2) + ast.to_bytes(3, "big")
        return hdr + self.payload

    @staticmethod
    def unpack(data: bytes) -> RtpPacket:
        if len(data) < 12 or data[0] >> 6 != 2:
            raise ValueError("not RTP v2")
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:12])
        cc = b0 & 0x0F
        off = 12 + 4 * cc
        ast = None
        if b0 & 0x10:  # header extension
            prof, ext_len = struct.unpack("!HH", data[off : off + 4])
            ext = data[off + 4 : off + 4 + 4 * ext_len]
            if prof == 0xBEDE:
                i = 0
                while i < len(ext):
                    if ext[i] == 0:
                        i += 1
                        continue
                    eid, ln = ext[i] >> 4, (ext[i] & 0xF) + 1
                    if eid == ABS_SEND_TIME_ID and ln == 3:
                        ast = int.from_bytes(ext[i + 1 : i + 4], "big")
                    i += 1 + ln
            off += 4 + 4 * ext_len
        end = len(data)
        if b0 & 0x20:  # padding
            end -= data[-1]
        return RtpPacket(b1 & 0x7F, seq, ts, ssrc, data[off:end], bool(b1 & 0x80), ast)


def is_rtcp(data: bytes) -> bool:
    return len(data) >= 2 and 192 <= data[1] <= 223


@dataclass
class ReportBlock:
    ssrc: int
    fraction_lost: int  # 0..255 (fraction * 256)
    cumulative_lost: int
    highest_seq: int
    jitter: int  # RTP timestamp units
    lsr: int
    dlsr: int  # 1/65536 s

    def pack(self) -> bytes:
        cl = self.cumulative_lost & 0xFFFFFF
        return struct.pack(
            "!IIIIII",
            self.ssrc,
            (self.fraction_lost << 24) | cl,
            self.highest_seq & 0xFFFFFFFF,
            self.jitter & 0xFFFFFFFF,
            self.lsr,
            self.dlsr,
        )

    @staticmethod
    def unpack(b: bytes) -> ReportBlock:
        ssrc, fl_cl, hs, jit, lsr, dlsr = struct.unpack("!IIIIII", b[:24])
        cl = fl_cl & 0xFFFFFF
        if cl & 0x800000:
            cl -= 1 << 24
        return ReportBlock(ssrc, fl_cl >> 24, cl, hs, jit, lsr, dlsr)


@dataclass
class SenderReport:
    ssrc: int
    ntp_msw: int
    ntp_lsw: int
    rtp_ts: int
    packets: int
    octets: int
    blocks: list[ReportBlock] = field(default_factory=list)


@dataclass
class ReceiverReport:
    ssrc: int
    blocks: list[ReportBlock] = field(default_factory=list)


@dataclass
class Pli:
    sender_ssrc: int
    media_ssrc: int


@dataclass
class Remb:
    sender_ssrc: int
    bitrate_bps: int
    ssrcs: list[int] = field(default_factory=list)


def _hdr(count: int, pt: int, body_len: int) -> bytes:
    assert body_len % 4 == 0
    return struct.pack("!BBH", 0x80 | (count & 0x1F), pt, body_len // 4)


def pack_sr(sr: SenderReport) -> bytes:
    body = struct.pack(
        "!IIIIII", sr.ssrc, sr.ntp_msw, sr.ntp_lsw, sr.rtp_ts & 0xFFFFFFFF, sr.packets, sr.octets
    )
    body += b"".join(b.pack() for b in sr.blocks)
    return _hdr(len(sr.blocks), 200, len(body)) + body


def pack_rr(rr: ReceiverReport) -> bytes:
    body = struct.pack("!I", rr.ssrc) + b"".join(b.pack() for b in rr.blocks)
    return _hdr(len(rr.blocks), 201, len(body)) + body


def pack_pli(p: Pli) -> bytes:
    body = struct.pack("!II", p.sender_ssrc, p.media_ssrc)
    return _hdr(1, 206, len(body)) + body


def pack_remb(r: Remb) -> bytes:
    br = max(0, int(r.bitrate_bps))
    exp = 0
    while br >= (1 << 18):
        br >>= 1
        exp += 1
    body = struct.pack("!II", r.sender_ssrc, 0) + b"REMB"
    body += struct.pack("!BBH", len(r.ssrcs), (exp << 2) | (br >> 16), br & 0xFFFF)
    body += b"".join(struct.pack("!I", s) for s in r.ssrcs)
    return _hdr(15, 206, len(body)) + body


def unpack_rtcp(data: bytes) -> list:
    """Parse a (compound) RTCP packet into a list of SR/RR/Pli/Remb objects (unknown types skipped)."""
    out: list = []
    off = 0
    while off + 4 <= len(data):
        b0, pt, length = struct.unpack("!BBH", data[off : off + 4])
        count = b0 & 0x1F
        end = off + 4 + 4 * length
        body = data[off + 4 : end]
        if pt == 200:
            ssrc, msw, lsw, ts, pk, oc = struct.unpack("!IIIIII", body[:24])
            blocks = [ReportBlock.unpack(body[24 + 24 * i :]) for i in range(count)]
            out.append(SenderReport(ssrc, msw, lsw, ts, pk, oc, blocks))
        elif pt == 201:
            (ssrc,) = struct.unpack("!I", body[:4])
            out.append(ReceiverReport(ssrc, [ReportBlock.unpack(body[4 + 24 * i :]) for i in range(count)]))
        elif pt == 206 and count == 1:
            s, m = struct.unpack("!II", body[:8])
            out.append(Pli(s, m))
        elif pt == 206 and count == 15 and body[8:12] == b"REMB":
            s = struct.unpack("!I", body[:4])[0]
            n, b1, lo = struct.unpack("!BBH", body[12:16])
            exp, mant = b1 >> 2, ((b1 & 0x3) << 16) | lo
            ssrcs = [struct.unpack("!I", body[16 + 4 * i : 20 + 4 * i])[0] for i in range(n)]
            out.append(Remb(s, mant << exp, ssrcs))
        elif pt == 205 and count == 1:
            out.append(_unpack_nack(body))
        elif pt == 204:
            app = unpack_app(body)
            if app:
                out.append(app)
        off = end
    return out


class ReceiverStats:
    """Per-source receive statistics for RR generation (RFC 3550 A.3, A.8)."""

    def __init__(self, ssrc: int, clock_rate: int):
        self.ssrc = ssrc
        self.clock = clock_rate
        self.base_seq: int | None = None
        self.max_seq = 0
        self.cycles = 0
        self.received = 0
        self.expected_prior = 0
        self.received_prior = 0
        self.jitter = 0.0
        self.transit: float | None = None
        self.last_sr_mid: int = 0
        self.last_sr_arrival: float = 0.0

    def on_packet(self, seq: int, ts: int, arrival: float) -> None:
        if self.base_seq is None:
            self.base_seq = seq
            self.max_seq = seq
        else:
            delta = (seq - self.max_seq) & 0xFFFF
            if 0 < delta < 0x8000:
                if seq < self.max_seq:
                    self.cycles += 1 << 16
                self.max_seq = seq
        self.received += 1
        transit = arrival * self.clock - ts
        if self.transit is not None:
            d = abs(transit - self.transit)
            self.jitter += (d - self.jitter) / 16.0
        self.transit = transit

    def on_sr(self, sr: SenderReport, arrival: float) -> None:
        self.last_sr_mid = ntp_mid32(sr.ntp_msw, sr.ntp_lsw)
        self.last_sr_arrival = arrival

    def report_block(self, now: float) -> ReportBlock:
        ext_max = self.cycles + self.max_seq
        expected = ext_max - (self.base_seq or 0) + 1 if self.base_seq is not None else 0
        lost = expected - self.received
        exp_int = expected - self.expected_prior
        rec_int = self.received - self.received_prior
        self.expected_prior, self.received_prior = expected, self.received
        lost_int = exp_int - rec_int
        frac = 0 if exp_int <= 0 or lost_int <= 0 else min(255, (lost_int << 8) // exp_int)
        dlsr = int((now - self.last_sr_arrival) * 65536) if self.last_sr_mid else 0
        return ReportBlock(self.ssrc, frac, max(lost, 0), ext_max, int(self.jitter), self.last_sr_mid, dlsr)


def rtt_from_block(b: ReportBlock, now: float) -> float | None:
    """Sender-side RTT (seconds) from a report block: A - LSR - DLSR."""
    if b.lsr == 0:
        return None
    msw, lsw = ntp_now(now)
    a = ntp_mid32(msw, lsw)
    rtt = ((a - b.lsr - b.dlsr) & 0xFFFFFFFF) / 65536.0
    return rtt if rtt < 60 else None


@dataclass
class AppFeedback:
    """RTCP APP (PT=204, name 'AEGS'): receiver-side inference feedback for the joint controller."""

    ssrc: int
    service_fps_milli: int
    queue_depth: int


def pack_app(a: AppFeedback) -> bytes:
    body = struct.pack("!I", a.ssrc) + b"AEGS" + struct.pack("!II", a.service_fps_milli, a.queue_depth)
    return _hdr(0, 204, len(body)) + body


def unpack_app(body: bytes) -> AppFeedback | None:
    if body[4:8] != b"AEGS":
        return None
    ssrc = struct.unpack("!I", body[:4])[0]
    fps, qd = struct.unpack("!II", body[8:16])
    return AppFeedback(ssrc, fps, qd)


def abs_send_time(t: float) -> int:
    """6.18 fixed-point seconds, wrapping every 64 s."""
    return int(t * (1 << 18)) & 0xFFFFFF


def ast_delta(arrival: float, ast: int) -> float:
    """arrival - send time (seconds) modulo the 64 s wrap; meaningful only as relative values."""
    d = (abs_send_time(arrival) - ast) & 0xFFFFFF
    return d / (1 << 18)


@dataclass
class Nack:
    """Generic NACK (RFC 4585 §6.2.1, RTPFB FMT=1)."""

    sender_ssrc: int
    media_ssrc: int
    seqs: list[int]


def pack_nack(n: Nack) -> bytes:
    seqs = sorted(set(s & 0xFFFF for s in n.seqs))
    fcis = []
    i = 0
    while i < len(seqs):
        pid = seqs[i]
        blp = 0
        j = i + 1
        while j < len(seqs) and 0 < ((seqs[j] - pid) & 0xFFFF) <= 16:
            blp |= 1 << (((seqs[j] - pid) & 0xFFFF) - 1)
            j += 1
        fcis.append(struct.pack("!HH", pid, blp))
        i = j
    body = struct.pack("!II", n.sender_ssrc, n.media_ssrc) + b"".join(fcis)
    return _hdr(1, 205, len(body)) + body


def _unpack_nack(body: bytes) -> Nack:
    s, m = struct.unpack("!II", body[:8])
    seqs = []
    for off in range(8, len(body), 4):
        pid, blp = struct.unpack("!HH", body[off : off + 4])
        seqs.append(pid)
        seqs += [(pid + k + 1) & 0xFFFF for k in range(16) if blp & (1 << k)]
    return Nack(s, m, seqs)
