"""Integration tests: Python CheckpointClient against the Rust aegis-ckpt server (P1 + P2)."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import blake3
import pytest
import torch

from ckpt.client import CheckpointClient
from ckpt.crypto import CkptCrypto, KeyService
from ckpt.errors import IntegrityError, NotFoundError, RollbackError, SignatureError
from ckpt.server import ServerProcess, find_binary
from ckpt.wire import Conn

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def binary() -> str:
    try:
        return find_binary()
    except FileNotFoundError:
        env = dict(os.environ)
        if sys.platform == "darwin":  # bindgen (librocksdb-sys) needs libclang
            lib = "/Library/Developer/CommandLineTools/usr/lib"
            env.setdefault("LIBCLANG_PATH", lib)
            env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", lib)
        subprocess.run(["cargo", "build", "--release", "-q"], cwd=ROOT / "ckpt", check=True, env=env)
        return find_binary()


@pytest.fixture
def server(tmp_path, binary):
    s = ServerProcess(tmp_path / "store", binary=binary).start()
    yield s
    s.stop()


def state(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "w": torch.randn(512, 512, generator=g),
        "b": torch.randn(512, generator=g).to(torch.bfloat16),
        "step": torch.tensor(7, dtype=torch.int64),
        "mask": torch.rand(100, generator=g) > 0.5,
        "empty": torch.empty(0, 3),
    }


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a
    )


def test_roundtrip_encrypted(server):
    cr = CkptCrypto.generate("job-a")
    with CheckpointClient(server.addr, cr, chunk_size=64 << 10) as c:
        st = state()
        s = c.save(st, step=3, extra={"lr": 0.1})
        assert s.seq == 1 and s.logical_bytes > 0 and s.uploaded_bytes > s.logical_bytes  # GCM overhead
        out, info = c.restore(trusted_hwm=c.hwm, trusted_anchor=c.anchor)
        assert same(st, out) and info.step == 3 and info.extra == {"lr": 0.1}


def test_roundtrip_plaintext(server):
    with CheckpointClient(server.addr, None, chunk_size=64 << 10) as c:
        st = state(1)
        c.save(st)
        out, _ = c.restore()
        assert same(st, out)


def test_server_never_sees_plaintext(server, tmp_path):
    cr = CkptCrypto.generate("job-a")
    marker = torch.full((4096,), 1234.5678, dtype=torch.float64)
    with CheckpointClient(server.addr, cr) as c:
        c.save({"m": marker})
    blob = b"".join(p.read_bytes() for p in (tmp_path / "store" / "segments").iterdir())
    assert marker.numpy().tobytes()[:64] not in blob


def test_dedup_across_incremental_saves(server):
    cr = CkptCrypto.generate("job-a")
    with CheckpointClient(server.addr, cr, chunk_size=64 << 10) as c:
        st = state()
        first = c.save(st)
        st["w"][:16] += 1.0  # touches only the first 64 KiB chunk of w
        second = c.save(st)
        assert second.dedup_hits == first.unique_chunks - 1
        assert second.uploaded_bytes < first.uploaded_bytes / 10
        out, _ = c.restore()
        assert same(st, out)
        assert c.list() == [1, 2]


def test_snapshot_is_async_and_isolated(server):
    with CheckpointClient(server.addr, CkptCrypto.generate("j")) as c:
        st = state()
        expect = {k: v.clone() for k, v in st.items()}
        h = c.snapshot(st)
        st["w"].zero_()  # mutate after snapshot: must not affect the checkpoint
        s = h.wait()
        assert s.stall_s < s.flush_s + 1.0
        out, _ = c.restore()
        assert same(expect, out)


def _segments(root: Path) -> list[Path]:
    return sorted((root / "segments").iterdir())


def test_tampered_chunk_rejected(server, tmp_path):
    with CheckpointClient(server.addr, CkptCrypto.generate("j")) as c:
        c.save(state())
        seg = _segments(tmp_path / "store")[0]
        data = bytearray(seg.read_bytes())
        data[72 + 100] ^= 0x01  # inside first chunk's payload
        seg.write_bytes(bytes(data))
        with pytest.raises(IntegrityError):
            c.restore()


def test_gcm_tamper_detected_client_side():
    cr = CkptCrypto.generate("j")
    cid = cr.chunk_id(b"secret")
    ct = bytearray(cr.encrypt_chunk(cid, b"secret"))
    assert cr.decrypt_chunk(cid, bytes(ct), {}) == b"secret"
    ct[-1] ^= 1
    with pytest.raises(IntegrityError):
        cr.decrypt_chunk(cid, bytes(ct), {})
    with pytest.raises(IntegrityError):  # ciphertext swapped onto another id
        cr.decrypt_chunk(cr.chunk_id(b"other"), cr.encrypt_chunk(cid, b"secret"), {})


def _rewrite_wal_manifest(root: Path, mutate) -> None:
    """Attacker with disk access: edit a manifest and recompute the WAL checksum (server can't tell)."""
    p = root / "wal" / "manifest.wal"
    data = p.read_bytes()
    out = b""
    pos = 0
    while pos < len(data):
        seq, ln = struct.unpack("<QI", data[pos + 4 : pos + 16])
        m = data[pos + 16 : pos + 16 + ln]
        m = mutate(seq, m)
        rec = b"AGW1" + struct.pack("<QI", seq, len(m)) + m
        out += rec + blake3.blake3(rec).digest()
        pos += 16 + ln + 32
    p.write_bytes(out)


def test_tampered_manifest_rejected(tmp_path, binary):
    root = tmp_path / "store"
    cr = CkptCrypto.generate("j")
    with ServerProcess(root, binary=binary) as s, CheckpointClient(s.addr, cr) as c:
        c.save(state(), step=10)

    def bump_step(seq, m):
        n = struct.unpack("<I", m[:4])[0]
        head, blob = m[: 4 + 32 * n], m[4 + 32 * n :]
        body = json.loads(blob[68:])
        body["step"] = 999
        return head + blob[:68] + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    _rewrite_wal_manifest(root, bump_step)
    with ServerProcess(root, binary=binary) as s, CheckpointClient(s.addr, cr) as c:
        with pytest.raises(SignatureError):
            c.restore()


def test_replayed_manifest_rejected(server):
    cr = CkptCrypto.generate("j")
    with CheckpointClient(server.addr, cr) as c:
        c.save(state(0))
        c.save(state(1))
        conn = Conn(server.addr)
        _, ids, blob1 = conn.get_manifest(1)
        conn.commit(3, ids, blob1)  # replay old signed manifest as the newest checkpoint
        with pytest.raises(SignatureError):
            c.restore()
        conn.close()


def test_rollback_rejected(tmp_path, binary):
    root, old = tmp_path / "store", tmp_path / "old"
    cr = CkptCrypto.generate("j")
    with ServerProcess(root, binary=binary) as s, CheckpointClient(s.addr, cr) as c:
        c.save(state(0))
        s.stop()
        shutil.copytree(root, old)  # attacker keeps a copy of the older store
    with ServerProcess(root, binary=binary) as s, CheckpointClient(s.addr, cr) as c:
        c.hwm = 1
        c.save(state(1))
        hwm, anchor = c.hwm, c.anchor
        assert hwm == 2
    with ServerProcess(old, binary=binary) as s, CheckpointClient(s.addr, cr) as c:
        with pytest.raises(RollbackError):
            c.restore(trusted_hwm=hwm)
        with pytest.raises(RollbackError):
            c.restore(trusted_anchor=anchor)


def test_wrong_keys_rejected(server):
    cr = CkptCrypto.generate("j")
    with CheckpointClient(server.addr, cr) as c:
        c.save(state())
    other_kms = CkptCrypto(job="j", kms=KeyService(), verify_key=cr.verify_key, mac_key=cr.mac_key)
    with CheckpointClient(server.addr, other_kms) as c, pytest.raises(IntegrityError):
        c.restore()
    other_sig = CkptCrypto.generate("j")
    with CheckpointClient(server.addr, other_sig) as c, pytest.raises(SignatureError):
        c.restore()


def test_dek_rotation_old_chunks_still_restore(server):
    cr = CkptCrypto.generate("j")
    with CheckpointClient(server.addr, cr, chunk_size=64 << 10) as c:
        st = state()
        c.save(st)
        cr.rotate_dek()
        st["w"][:16] += 1
        c.save(st)
        # restore with a fresh crypto object that only knows the KMS + pinned keys
        fresh = CkptCrypto(job="j", kms=cr.kms, verify_key=cr.verify_key, mac_key=cr.mac_key)
        with CheckpointClient(server.addr, fresh) as c2:
            out, _ = c2.restore()
        assert same(st, out)


def test_kill9_after_ack_is_durable(tmp_path, binary):
    root = tmp_path / "store"
    cr = CkptCrypto.generate("j")
    s = ServerProcess(root, binary=binary).start()
    c = CheckpointClient(s.addr, cr)
    st = state()
    c.save(st)
    s.kill9()
    c._ctl.close()
    s2 = ServerProcess(root, binary=binary).start()
    with CheckpointClient(s2.addr, cr) as c2:
        out, info = c2.restore(trusted_hwm=1)
        assert info.seq == 1 and same(st, out)
    s2.stop()


def test_gc_through_client(server):
    with CheckpointClient(server.addr, CkptCrypto.generate("j"), chunk_size=64 << 10) as c:
        st = state()
        for i in range(4):
            st["w"][i] += 1
            c.save(st)
        r = c.gc(2)
        assert r["manifests_retained"] == 2 and c.list() == [3, 4]
        out, _ = c.restore()
        assert same(st, out)
        with pytest.raises(NotFoundError):
            c.restore(seq=1)
        assert c.verify(3)["ok"]


def _flaky(monkeypatch, method: str, after_send: bool, failures: int = 1):
    """Make the first `failures` calls of Conn.<method> lose their ack (after_send) or the request."""
    real = getattr(Conn, method)
    calls = {"n": 0}

    def wrapper(self, *a, **k):
        calls["n"] += 1
        if calls["n"] <= failures:
            if after_send:
                real(self, *a, **k)
            raise ConnectionResetError(f"injected lost {'ack' if after_send else 'request'}")
        return real(self, *a, **k)

    monkeypatch.setattr(Conn, method, wrapper)
    return calls


@pytest.mark.parametrize("after_send", [False, True])
def test_lost_put_is_resent_before_commit(server, monkeypatch, after_send):
    # Liveness (TLA+ finding): a PUT whose ack is missing must be resent, else COMMIT stalls forever.
    st = state()
    with CheckpointClient(server.addr, CkptCrypto.generate("j"), chunk_size=64 << 10, retry_backoff=0.0) as c:
        calls = _flaky(monkeypatch, "put_chunks", after_send, failures=2)
        s = c.save(st)
        assert s.retries >= 2 and calls["n"] >= 3
        monkeypatch.undo()
        out, _ = c.restore()
        assert same(st, out)


@pytest.mark.parametrize("after_send", [False, True])
def test_lost_commit_ack_is_resolved(server, monkeypatch, after_send):
    st = state()
    with CheckpointClient(server.addr, CkptCrypto.generate("j"), retry_backoff=0.0) as c:
        _flaky(monkeypatch, "commit", after_send)
        s = c.save(st)
        monkeypatch.undo()
        assert s.seq == 1 and s.retries == 1 and c.list() == [1]
        out, _ = c.restore(trusted_hwm=1)
        assert same(st, out)
