"""EngineBackend: drop-in put/get compatibility with netlab.blobstore.Client for SliceCheckpointer."""

from __future__ import annotations

import os
import shutil
import threading

import pytest
import torch

from ckpt.client import EngineBackend
from ckpt.crypto import CkptCrypto
from ckpt.errors import IntegrityError, RollbackError
from ckpt.server import ServerProcess, find_binary
from train.checkpointing import SliceCheckpointer
from train.model import PRESETS, TinyVLM


@pytest.fixture
def srv(tmp_path):
    try:
        b = find_binary()
    except FileNotFoundError:
        pytest.skip("aegis-ckpt not built (make test-rust builds it)")
    s = ServerProcess(tmp_path / "store", binary=b).start()
    yield s
    s.stop()


def full_state():
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    torch.manual_seed(0)
    m = TinyVLM(PRESETS["tiny"])
    opt = torch.optim.AdamW(m.parameters())
    m.loss(torch.rand(2, 32, 32), torch.randint(1, 255, (2, 128))).backward()
    opt.step()
    msd, osd = get_state_dict(m, opt, options=StateDictOptions(full_state_dict=True))
    return {"model": msd, "opt": osd}


@pytest.mark.parametrize("encrypted", [False, True])
def test_slice_checkpointer_concurrent_ranks(srv, encrypted):
    """Three ranks save their slices concurrently to one server (shared manifest sequence)."""
    full = full_state()
    world = 3
    crypto = CkptCrypto.generate("j") if encrypted else None
    errs = []

    def rank(r):
        try:
            b = EngineBackend(srv.addr, job="j", rank=r, crypto=crypto)
            ck = SliceCheckpointer(b, r, world, job="j")
            for step in (10, 20):
                ck.save(step, full, blocking=True)
            b.close()
        except Exception as e:  # pragma: no cover - reported below
            errs.append(e)

    ts = [threading.Thread(target=rank, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    b = EngineBackend(srv.addr, job="j", crypto=crypto)
    got = SliceCheckpointer(b, 0, world, job="j").restore(20)
    for k, v in full["model"].items():
        assert torch.equal(got["model"][k], v)
    assert b.get("absent") is None
    assert sorted(b.cli.list()) == list(range(1, 7))


def test_put_get_incremental_and_rollback(srv, tmp_path):
    b = EngineBackend(srv.addr, chunk_size=4096)
    data = bytearray(os.urandom(1 << 20))
    b.put("k1", bytes(data))
    assert b.last_put.uploaded_bytes == 1 << 20
    data[5000:5010] = b"x" * 10
    b.put("k2", bytes(data))
    assert b.last_put.uploaded_bytes == 4096 and b.last_put.dedup_hits == 255
    assert b.get("k2") == bytes(data)
    assert b.get("") is None
    # another process (fresh object) finds keys by scanning manifests
    assert EngineBackend(srv.addr).get("k1")[:4999] == bytes(data[:4999])
    # a key re-put later wins; a stale server copy is rejected via the recorded seq
    b.put("k1", b"newer")
    assert b.get("k1") == b"newer"
    b.trusted["k1"] = 99
    with pytest.raises(RollbackError):
        b.get("k1")


@pytest.mark.netlab
def test_engine_backend_io_uring_in_lab(tmp_path):
    """Linux lab only: the io_uring backend end to end, plus kill -9 durability of an acked put."""
    s = ServerProcess(tmp_path / "u", io="uring").start()
    try:
        assert s.recovery is not None
        b = EngineBackend(s.addr, crypto=CkptCrypto.generate("lab"))
        assert b.cli.stats()["io"] == "uring"
        x = os.urandom(3 << 20)
        b.put("ckpt_lab_00000001_slice0of1", x)
        s.kill9()
        s.start()
        b2 = EngineBackend(s.addr, crypto=b.cli.crypto, trusted=b.trusted)
        assert b2.get("ckpt_lab_00000001_slice0of1") == x
    finally:
        s.stop()


def test_tampered_store_detected(srv):
    b = EngineBackend(srv.addr, chunk_size=4096)
    b.put("k", os.urandom(50_000))
    srv.stop()
    seg = sorted((srv.root / "segments").glob("*"))
    seg = max(seg, key=lambda p: p.stat().st_size)
    raw = bytearray(seg.read_bytes())
    raw[len(raw) // 2] ^= 1
    seg.write_bytes(bytes(raw))
    srv.start()
    with pytest.raises(IntegrityError):
        EngineBackend(srv.addr, chunk_size=4096).get("k")
    shutil.rmtree(srv.root, ignore_errors=True)


def test_keydir_lets_a_relaunched_job_restore(srv, tmp_path):
    """Migration/crash restart: new processes load the same job key dir and can verify + decrypt earlier
    checkpoints. A freshly generated key set (another job's material) cannot."""
    from ckpt.crypto import init_keydir, load_keydir

    kd = tmp_path / "keys"
    init_keydir(kd, "j")
    full = full_state()
    world = 2
    for r in range(world):
        b = EngineBackend(srv.addr, job="j", rank=r, crypto=load_keydir(kd, "j"))
        SliceCheckpointer(b, r, world, job="j").save(30, full, blocking=True)
        b.close()
    b2 = EngineBackend(srv.addr, job="j", rank=0, crypto=load_keydir(kd, "j"))  # "relaunched" process
    got = SliceCheckpointer(b2, 0, world, job="j").restore(30)
    k = next(iter(full["model"]))
    assert torch.equal(got["model"][k], full["model"][k])
    b2.close()
    other = EngineBackend(srv.addr, job="j", rank=0, crypto=CkptCrypto.generate("j"))
    with pytest.raises(Exception):  # noqa: B017 - signature or integrity error, whichever is hit first
        SliceCheckpointer(other, 0, world, job="j").restore(30)
    other.close()
    with pytest.raises(IntegrityError):
        load_keydir(kd, "another-job")
