"""aegis-ckpt engine through the Python client: drop-in blob interface, incrementality, crash recovery, GC."""

import os

import pytest
import torch

from ckpt import client as cc
from security import envelope as env
from train.checkpointing import SliceCheckpointer
from train.model import PRESETS, TinyVLM


def _state(seed=0):
    torch.manual_seed(seed)
    m = TinyVLM(PRESETS["tiny"])
    opt = torch.optim.AdamW(m.parameters())
    m.loss(torch.rand(2, 32, 32), torch.randint(1, 255, (2, 128))).backward()
    opt.step()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    msd, osd = get_state_dict(m, opt, options=StateDictOptions(full_state_dict=True))
    return {"model": msd, "opt": osd}


def test_stream_mapping():
    assert cc.default_stream("ckpt_job_00000020_slice1of6") == ("ckpt_job_slice1of6", 20)
    assert cc.default_stream("whatever") == ("whatever", -1)


def test_manifest_codec():
    ids = [os.urandom(32) for _ in range(3)]
    raw = cc.encode_manifest("k", "s", 7, ids, b"body")
    assert cc.parse_manifest(raw) == ("k", "s", 7, ids, b"body")
    with pytest.raises(env.BadManifest):
        cc.parse_manifest(raw[:-1])


def test_slice_checkpointer_roundtrip(engine):
    st = engine
    full = _state()
    world = 3
    for r in range(world):
        SliceCheckpointer(cc.EngineBackend(st.addr, job="j", rank=r), r, world, job="j").save(
            7, full, blocking=True
        )
    got = SliceCheckpointer(cc.EngineBackend(st.addr, job="j"), 0, world, job="j").restore(7)
    for k, v in full["model"].items():
        assert torch.equal(got["model"][k], v)
    assert cc.EngineBackend(st.addr).get("nope") is None


def test_incremental_put_sends_only_new_chunks(engine):
    st = engine
    b = cc.EngineBackend(st.addr, chunk_size=4096)
    data = bytearray(os.urandom(1 << 20))
    b.put("ckpt_j_00000001_slice0of1", bytes(data))
    assert b.last_put["new_chunks"] == 256
    b.put("ckpt_j_00000002_slice0of1", bytes(data))
    assert b.last_put["new_chunks"] == 0 and b.last_put["seq"] == 2
    data[5000:5010] = b"x" * 10  # touches one 4 KiB chunk
    b.put("ckpt_j_00000003_slice0of1", bytes(data))
    assert b.last_put["new_chunks"] == 1
    assert b.get("ckpt_j_00000003_slice0of1") == bytes(data)
    s = b.cli.stats()
    assert s["chunks_written"] == 257 and s["commits"] == 3


def test_kill9_recovers_last_committed(engine):
    trust = env.TrustStore()
    b = cc.EngineBackend(engine.addr, trust=trust)
    blobs = [os.urandom(300_000) for _ in range(3)]
    for i, x in enumerate(blobs):
        b.put(f"ckpt_j_{i:08d}_slice0of1", x)
    addr = engine.restart()
    b2 = cc.EngineBackend(addr, trust=trust)
    name, data = b2.restore_latest("ckpt_j_slice0of1")
    assert name == "ckpt_j_00000002_slice0of1" and data == blobs[2]
    assert b2.get("ckpt_j_00000000_slice0of1") == blobs[0]
    rec = b2.cli.recovery()
    assert rec["wal_records"] == 3 and rec["wal_error"] is None


def test_rollback_after_lost_commit_is_detected(engine):
    """The server loses its newest WAL record (e.g. bit rot in the tail): the trust store notices."""
    trust = env.TrustStore()
    b = cc.EngineBackend(engine.addr, trust=trust)
    for i in range(2):
        b.put(f"ckpt_j_{i:08d}_slice0of1", os.urandom(10_000))
    engine.kill()
    w = engine.root / "wal" / "manifest.wal"
    w.write_bytes(w.read_bytes()[:-3])  # damaged last record -> looks torn, truncated on recovery
    b2 = cc.EngineBackend(engine.start(), trust=trust)
    assert b2.cli.recovery()["wal_records"] == 1
    with pytest.raises(env.Rollback):
        b2.restore_latest("ckpt_j_slice0of1")
    with pytest.raises(env.Rollback):  # the writer refuses to fork history too
        b2.put("ckpt_j_00000005_slice0of1", b"x")
    # without trusted state the older checkpoint is all anyone could know about
    assert cc.EngineBackend(engine.addr).restore_latest("ckpt_j_slice0of1")[0] == "ckpt_j_00000000_slice0of1"


def test_gc_through_client(engine):
    st = engine
    b = cc.EngineBackend(st.addr, chunk_size=65536)
    base = os.urandom(1 << 20)
    for i in range(5):
        b.put(f"ckpt_j_{i:08d}_slice0of1", base + os.urandom(1 << 20))
    rep = b.cli.gc(keep=2, min_dead=0.0)
    assert rep["dropped_manifests"] == 3 and rep["deleted_segments"] >= 1
    assert b.get("ckpt_j_00000000_slice0of1") is None
    assert len(b.get("ckpt_j_00000004_slice0of1")) == 2 << 20
    assert b.cli.verify()["errors"] == []


@pytest.mark.netlab
def test_io_uring_backend_in_lab(tmp_path):
    p, addr, info = cc.start_server(tmp_path / "s", io="uring")
    try:
        b = cc.EngineBackend(addr)
        assert b.cli.stats()["io"] == "uring"
        x = os.urandom(3 << 20)
        b.put("k", x)
        assert b.get("k") == x
    finally:
        p.kill()
        p.wait()
