"""Protocol P2: verified recovery must reject every tampering / replay / rollback / wrong-key case.

The server is untrusted for confidentiality and authenticity, so most cases are injected by a
malicious-server wrapper around the real engine client (`Evil`), plus on-disk tampering of a real store.
"""

import base64
import json
import os

import pytest

from ckpt import client as cc
from security import envelope as env

JOB = "jobA"
KEY = "ckpt_jobA_00000003_slice0of1"
STREAM = "ckpt_jobA_slice0of1"


def make_env(ks=None, sk=None, epoch=1):
    ks = ks or env.KeyService()
    sk = sk or env.new_signing_key()
    return env.Envelope(JOB, ks, sk, key_epoch=epoch), ks, sk


# ------------------------------------------------------------------------------- primitives


def test_chunk_seal_open_and_tamper():
    e, ks, _ = make_env()
    k = e.writer_keys()
    pt = os.urandom(5000)
    cid, ct = env.seal_chunk(k, pt)
    assert env.open_chunk(k.dek, k.mac_key, cid, ct) == pt
    for pos in (0, 12, len(ct) // 2, len(ct) - 1):  # nonce, body, tag
        bad = bytearray(ct)
        bad[pos] ^= 1
        with pytest.raises(env.TamperedChunk):
            env.open_chunk(k.dek, k.mac_key, cid, bytes(bad))
    with pytest.raises(env.TamperedChunk):  # ciphertext moved to another id (aad binding)
        env.open_chunk(k.dek, k.mac_key, os.urandom(32), ct)
    # id is keyed: no plaintext hash leaks, and another epoch derives a different mac key
    assert cid != env.chunk_id(ks.mac_key(JOB, 2), pt)


def test_dek_wrap_bound_to_job_and_epoch():
    ks = env.KeyService()
    dek = os.urandom(32)
    w = ks.wrap(JOB, 1, dek)
    assert ks.unwrap(JOB, 1, w) == dek
    with pytest.raises(env.WrongKey):
        ks.unwrap(JOB, 2, w)
    with pytest.raises(env.WrongKey):
        ks.unwrap("other", 1, w)
    with pytest.raises(env.WrongKey):
        env.KeyService().unwrap(JOB, 1, w)


def test_manifest_signature():
    e, _, sk = make_env()
    m = e.sign(e.make_doc(STREAM, KEY, 1, None, 3, []))
    assert e.verify(m.body).doc["seq"] == 1
    bad = bytearray(m.body)
    bad[-5] ^= 1
    with pytest.raises(env.BadSignature):
        e.verify(bytes(bad))
    other, _, _ = make_env()
    with pytest.raises(env.BadSignature):
        other.verify(m.body)


def test_trust_store_chain_logic():
    e, _, _ = make_env()
    chain = []
    prev = None
    for s in range(1, 6):
        m = e.sign(e.make_doc(STREAM, f"k{s}", s, prev, s, []))
        chain.append(m)
        prev = m.hash

    def fetch(s):
        return chain[s - 1] if 1 <= s <= len(chain) else None

    t = env.TrustStore()
    for m in chain[:3]:
        t.record(STREAM, m.seq, m.hash, m.doc["name"])
    assert t.check(STREAM, chain[2], "k3", fetch, latest=True) == "known"
    assert t.check(STREAM, chain[4], "k5", fetch, latest=True) == "chain-forward"
    with pytest.raises(env.Rollback):
        t.check(STREAM, chain[1], "k2", fetch, latest=True)
    # a fork: validly signed but different content at a known seq
    fork = e.sign(e.make_doc(STREAM, "k3", 3, chain[1].hash, 99, []))
    with pytest.raises(env.Rollback):
        t.check(STREAM, fork, "k3", fetch)
    # a newer manifest that does not extend the trusted chain
    stray = e.sign(e.make_doc(STREAM, "k4", 4, fork.hash, 4, []))
    with pytest.raises(env.Rollback):
        t.check(STREAM, stray, "k4", fetch)
    # older-than-known, reached by walking back from a trusted newer one
    t2 = env.TrustStore()
    t2.record(STREAM, 5, chain[4].hash, "k5")
    assert t2.check(STREAM, chain[1], "k2", fetch) == "chain-back"
    with pytest.raises(env.Rollback):
        t2.check(STREAM, e.sign(e.make_doc(STREAM, "k2", 2, None, 0, [])), "k2", fetch)


def test_trust_store_persists(tmp_path):
    p = tmp_path / "trust.json"
    t = env.TrustStore(p)
    t.record(STREAM, 4, "ab", "n")
    assert env.TrustStore(p).hwm(STREAM) == (4, "ab")


# ------------------------------------------------------------------- against a real server


class Evil(cc.EngineClient):
    """A malicious server: rewrites replies according to `self.mode`."""

    def __init__(self, addr):
        super().__init__(addr)
        self.mode = None
        self.saved = {}

    def get_manifest(self, name):
        raw = super().get_manifest(name)
        if raw is None:
            return raw
        n, s, q, ids, body = cc.parse_manifest(raw)
        if self.mode == "manifest_body":
            b = bytearray(body)
            b[80] ^= 1
            return cc.encode_manifest(n, s, q, ids, bytes(b))
        if self.mode == "manifest_header":
            return cc.encode_manifest(n, s, q, ids[::-1], body)
        if self.mode == "replay":
            return self.saved["old"]
        if self.mode == "swap":
            return self.saved["other"]
        return raw

    def latest(self, stream=""):
        if self.mode == "replay":
            return self.saved["old"]
        return super().latest(stream)

    def get_chunks(self, ids):
        out = super().get_chunks(ids)
        if self.mode == "chunk":
            b = bytearray(out[0])
            b[20] ^= 0x40
            out[0] = bytes(b)
        if self.mode == "chunk_swap" and len(out) > 1:
            out[0], out[1] = out[1], out[0]
        return out


@pytest.fixture
def secure(engine):
    e, ks, sk = make_env()
    trust = env.TrustStore()
    b = cc.EngineBackend(engine.addr, job=JOB, envelope=e, trust=trust, chunk_size=4096)
    data = {s: os.urandom(20_000) for s in range(1, 4)}
    for s, d in data.items():
        b.put(f"ckpt_{JOB}_{s:08d}_slice0of1", d)
    evil = Evil(engine.addr)
    evil.saved["old"] = evil.get_manifest(f"ckpt_{JOB}_00000001_slice0of1")
    evil.saved["other"] = evil.get_manifest(f"ckpt_{JOB}_00000002_slice0of1")
    b.cli = evil
    return b, data, e, ks, sk, trust, engine


def test_secure_roundtrip_and_ciphertext_on_server(secure):
    b, data, *_ = secure
    b.cli.mode = None
    assert b.get(KEY) == data[3]
    name, latest = b.restore_latest(STREAM)
    assert (name, latest) == (KEY, data[3])
    # what the server stores is ciphertext, and the ids are not plain BLAKE3 of the plaintext
    import blake3

    raw = b.cli.get_manifest(KEY)
    ids = cc.parse_manifest(raw)[3]
    ct = b.cli.get_chunks(ids[:1])[0]
    assert data[3][:4096] not in ct and ids[0] != blake3.blake3(data[3][:4096]).digest()


@pytest.mark.parametrize(
    "mode,exc",
    [
        ("chunk", env.TamperedChunk),
        ("chunk_swap", env.TamperedChunk),
        ("manifest_body", env.BadSignature),
        ("manifest_header", env.BadManifest),
        ("swap", env.BadManifest),
    ],
)
def test_malicious_server_is_rejected(secure, mode, exc):
    b, *_ = secure
    b.cli.mode = mode
    with pytest.raises(exc):
        b.get(KEY)


def test_replayed_manifest_is_rejected(secure):
    b, *_ = secure
    b.cli.mode = "replay"
    with pytest.raises(env.Rollback):  # asking for the newest: an old (validly signed) one is replayed
        b.restore_latest(STREAM)
    with pytest.raises(env.Rollback):  # the writer must not continue from a rolled-back head either
        b.put(f"ckpt_{JOB}_00000004_slice0of1", b"x")


def test_replayed_commit_is_rejected_by_server(secure):
    b, *_ = secure
    b.cli.mode = None
    with pytest.raises(cc.EngineError, match="replay"):
        b.cli.commit(b.cli.saved["old"])


def test_rolled_back_store_is_rejected(secure, tmp_path):
    """Restore from an *older copy of the whole store* (snapshot rollback) -> hwm check fails."""
    b, data, e, ks, sk, trust, engine = secure
    import shutil

    engine.kill()
    snap = tmp_path / "snap"
    shutil.copytree(engine.root, snap)
    engine.start()
    b2 = cc.EngineBackend(engine.addr, job=JOB, envelope=e, trust=trust, chunk_size=4096)
    b2.put(f"ckpt_{JOB}_00000004_slice0of1", os.urandom(1000))
    engine.kill()
    shutil.rmtree(engine.root)
    shutil.copytree(snap, engine.root)
    b3 = cc.EngineBackend(engine.start(), job=JOB, envelope=e, trust=trust, chunk_size=4096)
    with pytest.raises(env.Rollback):
        b3.restore_latest(STREAM)


def test_wrong_keys_are_rejected(secure):
    b, data, e, ks, sk, trust, engine = secure
    b.cli.mode = None
    wrong_kek = env.Envelope(JOB, env.KeyService(), sk, key_epoch=1)
    with pytest.raises(env.WrongKey):
        cc.EngineBackend(engine.addr, job=JOB, envelope=wrong_kek, trust=env.TrustStore()).get(KEY)
    wrong_pin = env.Envelope(JOB, ks, verify_key=env.new_signing_key().public_key())
    with pytest.raises(env.BadSignature):
        cc.EngineBackend(engine.addr, job=JOB, envelope=wrong_pin, trust=env.TrustStore()).get(KEY)
    other_job = env.Envelope("jobB", ks, sk)
    with pytest.raises(env.BadManifest):
        cc.EngineBackend(engine.addr, job="jobB", envelope=other_job, trust=env.TrustStore()).get(KEY)
    # a manifest whose wrapped DEK was swapped for another epoch's is rejected at unwrap
    raw = b.cli.get_manifest(KEY)
    n, s, q, ids, body = cc.parse_manifest(raw)
    doc = json.loads(body[env.SIG :])
    doc["wrapped_dek"] = base64.b64encode(ks.wrap(JOB, 9, os.urandom(32))).decode()
    resigned = e.sign(doc)  # even with a valid signature (compromised writer), epoch binding holds
    with pytest.raises(env.WrongKey):
        e.reader_keys(resigned)


def test_on_disk_tamper_detected_by_server_or_client(secure):
    b, data, e, ks, sk, trust, engine = secure
    engine.kill()
    seg = sorted((engine.root / "segments").glob("seg-*.log"))[0]
    raw = bytearray(seg.read_bytes())
    raw[200] ^= 1
    seg.write_bytes(bytes(raw))
    b2 = cc.EngineBackend(engine.start(), job=JOB, envelope=e, trust=trust, chunk_size=4096)
    with pytest.raises((cc.EngineError, env.EnvelopeError)):
        b2.get(f"ckpt_{JOB}_00000001_slice0of1")


def test_key_rotation_reencrypts_and_old_checkpoints_stay_readable(secure):
    b, data, e, ks, sk, trust, engine = secure
    b.cli.mode = None
    e.rotate(2)
    b.put(f"ckpt_{JOB}_00000004_slice0of1", data[3])  # same plaintext, new epoch
    assert b.last_put["new_chunks"] == b.last_put["chunks"]  # no cross-epoch dedup
    b.put(f"ckpt_{JOB}_00000005_slice0of1", data[3])
    assert b.last_put["new_chunks"] == 0  # dedup within the new epoch
    assert b.get(f"ckpt_{JOB}_00000001_slice0of1") == data[1]  # epoch-1 DEK still unwraps
