import numpy as np
import pytest

from media import rtp as R
from media.adapt import LADDER_KBPS, Decision, DelayBasedEstimator, IndependentPolicy, JointPolicy, pick_rung
from media.av1rtp import Depacketizer, leb128_decode, leb128_encode, packetize, parse_obus
from media.inference import Accelerator, FrameSelector, SimulatedStt, chart_point, intent_of
from media.netem import LinkProfile, _Dir

av = pytest.importorskip("av")


def test_leb128_roundtrip():
    for n in [0, 1, 127, 128, 300, 2**20 + 5]:
        b = leb128_encode(n)
        assert leb128_decode(b, 0) == (n, len(b))


def test_rtp_roundtrip_with_abs_send_time():
    p = R.RtpPacket(96, 65535, 123456, 0xABC, b"hello", True, R.abs_send_time(12.5))
    q = R.RtpPacket.unpack(p.pack())
    assert (q.pt, q.seq, q.ts, q.ssrc, q.payload, q.marker, q.abs_send_time) == (
        96,
        65535,
        123456,
        0xABC,
        b"hello",
        True,
        p.abs_send_time,
    )


def test_rtcp_compound_roundtrip():
    rb = R.ReportBlock(0x1111, 25, 7, 70000, 90, 1234, 5678)
    pkt = (
        R.pack_rr(R.ReceiverReport(0x3333, [rb]))
        + R.pack_remb(R.Remb(0x3333, 1_234_567, [0x1111]))
        + R.pack_pli(R.Pli(0x3333, 0x1111))
        + R.pack_nack(R.Nack(0x3333, 0x1111, [10, 12, 26, 40]))
        + R.pack_app(R.AppFeedback(0x3333, 6500, 2))
    )
    out = R.unpack_rtcp(pkt)
    assert out[0].blocks[0] == rb
    assert abs(out[1].bitrate_bps - 1_234_567) / 1_234_567 < 1e-4
    assert isinstance(out[2], R.Pli)
    assert sorted(out[3].seqs) == [10, 12, 26, 40]
    assert out[4].service_fps_milli == 6500


def test_receiver_stats_loss_fraction():
    st = R.ReceiverStats(1, 90000)
    for s in range(100):
        if s % 10 != 3:
            st.on_packet(s, s * 3000, s / 30)
    b = st.report_block(10.0)
    assert b.cumulative_lost == 10 and abs(b.fraction_lost / 256 - 0.10) < 0.01


def _encode_tus(n=12, kbps=300):
    from media.codec import Av1Encoder, EncoderSettings
    from media.source import build_source

    src = build_source(0, duration_s=n / 30, use_say=False)
    enc = Av1Encoder(256, 256, EncoderSettings(kbps, threads=2))
    tus = []
    for i, f in enumerate(src.frames):
        tus += [tu for tu, _ in enc.encode(f, i * 3000)]
    return src, tus


def test_av1_rtp_roundtrip_decodes():
    from media.codec import Av1Decoder

    src, tus = _encode_tus()
    dec = Av1Decoder()
    n = 0
    for tu in tus:
        pk = packetize(tu, mtu_payload=300)  # small MTU forces fragmentation (Z/Y bits)
        assert pk[-1][1] and not any(m for _, m in pk[:-1])
        out = Depacketizer.assemble([p for p, _ in pk])
        # identical OBU content after TD removal / size-field rewrite
        a = [(o.obu_type, o.payload) for o in parse_obus(tu) if o.obu_type != 2]
        b = [(o.obu_type, o.payload) for o in parse_obus(out) if o.obu_type != 2]
        assert a == b
        n += len(dec.decode(out))
    assert n == len(tus)
    assert packetize(tus[0])[0][0][0] & 0x08  # N bit on first packet of the keyframe TU


def test_depacketizer_rejects_missing_fragment():
    _, tus = _encode_tus(n=2)
    pk = packetize(tus[0], mtu_payload=200)
    assert len(pk) > 2
    with pytest.raises(ValueError):
        Depacketizer.assemble([pk[0][0], pk[2][0]])


def test_netem_seeded_and_loss_rate():
    p = LinkProfile("t", delay_ms=10, ge_p=0.0155, ge_r=0.5)
    runs = []
    for _ in range(2):
        d = _Dir(p, np.random.default_rng(3), 0.0)
        runs.append([d.schedule(100, i * 0.001) for i in range(20000)])
    assert runs[0] == runs[1]
    loss = sum(x is None for x in runs[0]) / len(runs[0])
    assert abs(loss - p.mean_loss()) < 0.01


def test_netem_rate_limit_queue_drop():
    d = _Dir(LinkProfile("r", delay_ms=0, rate_kbps=100, queue_ms=50), np.random.default_rng(0), 0.0)
    out = [d.schedule(1250, 0.0) for _ in range(20)]  # 100 ms each at 100 kbps
    assert out[0] == pytest.approx(0.1)
    assert sum(x is None for x in out) >= 18


def test_delay_estimator_detects_queue_growth():
    e = DelayBasedEstimator()
    for i in range(60):
        t = i / 30
        e.on_packet(t, 1200, 0.03 + 0.004 * i)  # one-way delay growing 120 ms/s
        e.update(t)
    assert e.state == "overuse"


def test_pick_rung_and_policies():
    assert pick_rung(1000, 400, 2.0) == 700  # one rung up after hold time
    assert pick_rung(1000, 400, 0.5) == 400  # hold
    assert pick_rung(150, 400, 0.1) == 100  # immediate down
    q = {f"{b}@{f}": float(f == 10) for b in LADDER_KBPS for f in (30, 15, 10)}
    d = JointPolicy(q).decide(1000, Decision(400, 30, 10, "latest"), 2.0, service_fps=5.0)
    assert d.fps == 10 and d.select_fps == pytest.approx(4.0) and d.queue_policy == "latest"
    assert IndependentPolicy().decide(1000, Decision(400, 30, 10, "fifo"), 2.0, 5.0).fps == 30


def test_chart_reader_and_selector():
    from media.source import build_source

    src = build_source(1, duration_s=2.0, use_say=False)
    x, y = chart_point(src.frames[-1])
    assert x > 0.9 and 0.5 <= y <= 1.0  # chart lives in the bottom half
    sel = FrameSelector(max_fps=10)
    picks = sum(sel.offer(f, i / 30) for i, f in enumerate(src.frames))
    assert 1 <= picks <= 21


def test_intent():
    assert intent_of("And what's my closing balance?") == "balance"
    assert intent_of("how much I spent on dining") == "spending"
    assert intent_of("flag the largest charge") == "largest"


class _FakeVision:
    def __call__(self, rgb):
        import time

        time.sleep(0.02)
        return {"chart_x": 0.0, "chart_y": 0.0}


@pytest.mark.parametrize("policy", ["fifo", "latest"])
def test_accelerator_backpressure_never_blocks(policy):
    import time

    done = []
    acc = Accelerator(_FakeVision(), SimulatedStt(0.0), policy, lambda j, r, t0, t1: done.append(j.kind))
    t = time.monotonic()
    for i in range(50):
        acc.submit_frame({"rgb": None, "idx": i})
    acc.submit_stt({"pcm16k": np.zeros(160, np.float32), "oracle": "closing balance"})
    assert time.monotonic() - t < 0.05  # media path not blocked
    time.sleep(0.6)
    acc.stop()
    assert acc.dropped_frames > 0 and "stt" in done
    if policy == "latest":  # STT preempts queued vision work
        assert done.index("stt") <= 2
