"""Checkpoint confidentiality and authenticity, client side (protocol P2, docs/protocols.md).

Primitives come from `cryptography` (AES-256-GCM, HKDF-SHA256, Ed25519, SHA-256); keyed chunk ids use
BLAKE3 in keyed mode from the `blake3` package, as P2 specifies. No primitive is implemented here.

* KEK: held by `KeyService` (in the lab it stands in for a KMS/HSM inside the security service).
* DEK: per (job, key epoch) AES-256-GCM key; stored in every manifest *wrapped* by the KEK:
  `wrapped = nonce(12) || AES-GCM(KEK, nonce, DEK, aad="aegis-dek|v1|<job>|<epoch>")`.
* chunk_id = BLAKE3-keyed(mac_key(job, epoch), plaintext); mac_key = HKDF-SHA256(KEK, info=job|epoch).
  Dedup works across all checkpoints of one key epoch; a DEK rotation (new epoch) re-encrypts every
  chunk once, because the old ciphertext is bound to the old DEK.
* ciphertext = nonce(12) || AES-GCM(DEK, nonce, plaintext, aad=chunk_id).
* Manifest = canonical JSON {v, job, stream, name, seq, prev_manifest_hash, step, created_at, key_epoch,
  wrapped_dek, shards:[{name, chunks:[id..], len}]}, Ed25519-signed with the job signing key; the
  verifier pins the public key. manifest_hash = SHA-256(signature || json).
* Restore checks: signature; the server's cleartext header equals the signed content; `seq` against
  the trusted high-water mark and the `prev_manifest_hash` chain (`TrustStore`); every chunk's GCM
  tag; chunk_id recomputed from the plaintext. Any failure raises an `EnvelopeError` subclass.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import blake3
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

NONCE = 12
SIG = 64


class EnvelopeError(Exception):
    """Base class: verified recovery rejected the checkpoint."""


class TamperedChunk(EnvelopeError):
    pass


class BadSignature(EnvelopeError):
    pass


class BadManifest(EnvelopeError):
    pass


class Rollback(EnvelopeError):
    """Replayed, rolled-back or forked manifest (seq / hash-chain check failed)."""


class WrongKey(EnvelopeError):
    pass


# ------------------------------------------------------------------------------------------ keys


class KeyService:
    """Holds the KEK. Wrap/unwrap DEKs and derive per-epoch chunk-MAC keys. Never exports the KEK."""

    def __init__(self, kek: bytes | None = None):
        self._kek = kek if kek is not None else AESGCM.generate_key(bit_length=256)
        if len(self._kek) != 32:
            raise ValueError("KEK must be 32 bytes")

    @staticmethod
    def _aad(job: str, epoch: int) -> bytes:
        return f"aegis-dek|v1|{job}|{epoch}".encode()

    def wrap(self, job: str, epoch: int, dek: bytes) -> bytes:
        n = os.urandom(NONCE)
        return n + AESGCM(self._kek).encrypt(n, dek, self._aad(job, epoch))

    def unwrap(self, job: str, epoch: int, wrapped: bytes) -> bytes:
        try:
            return AESGCM(self._kek).decrypt(wrapped[:NONCE], wrapped[NONCE:], self._aad(job, epoch))
        except (InvalidTag, ValueError) as e:
            raise WrongKey(f"cannot unwrap DEK for job {job!r} epoch {epoch}") from e

    def mac_key(self, job: str, epoch: int) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=f"aegis-chunk-mac|v1|{job}|{epoch}".encode()
        ).derive(self._kek)


@dataclass
class JobKeys:
    """Writer-side key material for one job and key epoch."""

    job: str
    epoch: int
    dek: bytes
    wrapped_dek: bytes
    mac_key: bytes

    @classmethod
    def new(cls, ks: KeyService, job: str, epoch: int) -> JobKeys:
        dek = AESGCM.generate_key(bit_length=256)
        return cls(job, epoch, dek, ks.wrap(job, epoch, dek), ks.mac_key(job, epoch))

    @classmethod
    def adopt(cls, ks: KeyService, job: str, epoch: int, wrapped: bytes) -> JobKeys:
        return cls(job, epoch, ks.unwrap(job, epoch, wrapped), wrapped, ks.mac_key(job, epoch))


# ---------------------------------------------------------------------------------------- chunks


def chunk_id(mac_key: bytes, plaintext: bytes) -> bytes:
    return blake3.blake3(plaintext, key=mac_key).digest()


def encrypt_chunk(keys: JobKeys, cid: bytes, plaintext: bytes) -> bytes:
    """nonce(12) || AES-GCM(DEK, nonce, plaintext, aad=chunk_id). Random 96-bit nonces: keep well under
    2^32 encryptions per DEK (NIST SP 800-38D); with dedup each distinct chunk is encrypted once per epoch."""
    n = os.urandom(NONCE)
    return n + AESGCM(keys.dek).encrypt(n, plaintext, cid)


def seal_chunk(keys: JobKeys, plaintext: bytes) -> tuple[bytes, bytes]:
    """Returns (chunk_id, nonce || ciphertext+tag)."""
    cid = chunk_id(keys.mac_key, plaintext)
    return cid, encrypt_chunk(keys, cid, plaintext)


def open_chunk(dek: bytes, mac_key: bytes, cid: bytes, ct: bytes) -> bytes:
    if len(ct) < NONCE + 16:
        raise TamperedChunk(f"chunk {cid.hex()[:16]}: ciphertext too short")
    try:
        pt = AESGCM(dek).decrypt(ct[:NONCE], ct[NONCE:], cid)
    except InvalidTag as e:
        raise TamperedChunk(f"chunk {cid.hex()[:16]}: AES-GCM tag mismatch") from e
    if chunk_id(mac_key, pt) != cid:
        raise TamperedChunk(f"chunk {cid.hex()[:16]}: chunk id does not match plaintext")
    return pt


# ------------------------------------------------------------------------------------- manifests


def canonical(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def manifest_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


@dataclass
class Manifest:
    doc: dict
    body: bytes  # what the server stores as the opaque manifest body
    hash: str

    @property
    def seq(self) -> int:
        return int(self.doc["seq"])

    @property
    def prev(self) -> str | None:
        return self.doc.get("prev_manifest_hash")

    @property
    def chunk_ids(self) -> list[bytes]:
        return [bytes.fromhex(c) for s in self.doc["shards"] for c in s["chunks"]]


def sign_manifest(sk: Ed25519PrivateKey, doc: dict) -> Manifest:
    js = canonical(doc)
    body = sk.sign(js) + js
    return Manifest(doc, body, manifest_hash(body))


def verify_manifest(pk: Ed25519PublicKey, body: bytes, job: str) -> Manifest:
    if len(body) <= SIG:
        raise BadManifest("manifest body too short")
    sig, js = body[:SIG], body[SIG:]
    try:
        pk.verify(sig, js)
    except InvalidSignature as e:
        raise BadSignature("manifest signature invalid (tampered manifest or wrong signing key)") from e
    doc = json.loads(js)
    if doc.get("job") != job:
        raise BadManifest(f"manifest is for job {doc.get('job')!r}, expected {job!r}")
    return Manifest(doc, body, manifest_hash(body))


def unsigned_manifest(doc: dict) -> Manifest:
    """Plaintext mode: no signature, integrity only (hash chain + server-side checksums)."""
    body = canonical(doc)
    return Manifest(doc, body, manifest_hash(body))


def parse_unsigned(body: bytes) -> Manifest:
    try:
        doc = json.loads(body)
    except ValueError as e:
        raise BadManifest("unparseable manifest") from e
    return Manifest(doc, body, manifest_hash(body))


# ----------------------------------------------------------------------------------- trust store


@dataclass
class TrustStore:
    """Trusted per-stream high-water marks and the manifest hashes the writer committed.

    P2 places this at the controller; here it is a JSON file (or memory) shared by writer and restorer.
    `fetch(stream, seq)` must return the *verified* manifest with that seq (or None) for chain walks.
    """

    path: Path | None = None
    streams: dict = field(default_factory=dict)  # stream -> {"hwm": int, "seqs": {str(seq): hash}}
    names: dict = field(default_factory=dict)  # name -> newest seq committed under that name
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.exists():
                d = json.loads(self.path.read_text())
                self.streams, self.names = d.get("streams", {}), d.get("names", {})

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"streams": self.streams, "names": self.names}))
        os.replace(tmp, self.path)

    def reload(self) -> None:
        if self.path is not None and self.path.exists():
            with self.lock:
                d = json.loads(self.path.read_text())
                self.streams, self.names = d.get("streams", {}), d.get("names", {})

    def hwm(self, stream: str) -> tuple[int, str | None]:
        with self.lock:
            st = self.streams.get(stream)
            if not st:
                return 0, None
            return st["hwm"], st["seqs"].get(str(st["hwm"]))

    def record(self, stream: str, seq: int, h: str, name: str, keep: int = 64) -> None:
        with self.lock:
            st = self.streams.setdefault(stream, {"hwm": 0, "seqs": {}})
            st["seqs"][str(seq)] = h
            st["hwm"] = max(st["hwm"], seq)
            if len(st["seqs"]) > keep:
                for s in sorted(st["seqs"], key=int)[:-keep]:
                    del st["seqs"][s]
            self.names[name] = max(self.names.get(name, 0), seq)
            self._save()

    def check(
        self,
        stream: str,
        m: Manifest,
        name: str,
        fetch: Callable[[int], Manifest | None],
        latest: bool = False,
    ) -> str:
        """Raise Rollback unless `m` is on the trusted chain. Returns how it was established."""
        with self.lock:
            st = self.streams.get(stream)
            known = {int(k): v for k, v in st["seqs"].items()} if st else {}
            hwm = st["hwm"] if st else 0
            name_seq = self.names.get(name, 0)
        if m.seq < name_seq:
            raise Rollback(f"{name}: manifest seq {m.seq} older than committed seq {name_seq} (rollback)")
        if latest and m.seq < hwm:
            raise Rollback(f"stream {stream}: latest manifest seq {m.seq} < high-water mark {hwm}")
        if not st:
            return "tofu"  # no trusted state for this stream: nothing to compare against
        if m.seq in known:
            if known[m.seq] != m.hash:
                raise Rollback(f"stream {stream}: manifest seq {m.seq} differs from the committed one")
            return "known"
        above = [s for s in known if s > m.seq]
        if above:
            # walk back from the nearest trusted newer manifest to m.seq
            a = min(above)
            cur = fetch(a)
            if cur is None or cur.hash != known[a]:
                raise Rollback(f"stream {stream}: cannot re-establish trusted manifest seq {a}")
            while cur.seq > m.seq + 1:
                nxt = fetch(cur.seq - 1)
                if nxt is None or nxt.hash != cur.prev:
                    raise Rollback(f"stream {stream}: hash chain broken at seq {cur.seq - 1}")
                cur = nxt
            if cur.prev != m.hash:
                raise Rollback(f"stream {stream}: manifest seq {m.seq} is not on the trusted chain")
            return "chain-back"
        # newer than anything trusted: walk back from m to the newest trusted seq (or genesis)
        below = max(known) if known else 0
        cur = m
        while cur.seq > below + 1:
            nxt = fetch(cur.seq - 1)
            if nxt is None or nxt.hash != cur.prev:
                raise Rollback(f"stream {stream}: hash chain broken below seq {cur.seq}")
            cur = nxt
        want = known.get(below) if below else None
        if cur.prev != want:
            raise Rollback(f"stream {stream}: manifest seq {m.seq} does not extend the trusted chain")
        return "chain-forward"


# ------------------------------------------------------------------------------- writer / reader


class Envelope:
    """Bundles the key service, signing key / pinned verify key and the job's current key epoch."""

    def __init__(
        self,
        job: str,
        key_service: KeyService,
        signing_key: Ed25519PrivateKey | None = None,
        verify_key: Ed25519PublicKey | None = None,
        key_epoch: int = 1,
    ):
        self.job, self.ks, self.sk, self.epoch = job, key_service, signing_key, key_epoch
        self.pk = verify_key or (signing_key.public_key() if signing_key else None)
        if self.pk is None:
            raise ValueError("need a signing key or a pinned verify key")
        self._keys: JobKeys | None = None
        self._deks: dict[tuple[int, bytes], JobKeys] = {}

    def writer_keys(self, adopt_wrapped: tuple[int, bytes] | None = None) -> JobKeys:
        if self._keys is None or self._keys.epoch != self.epoch:
            if adopt_wrapped and adopt_wrapped[0] == self.epoch:
                self._keys = JobKeys.adopt(self.ks, self.job, self.epoch, adopt_wrapped[1])
            else:
                self._keys = JobKeys.new(self.ks, self.job, self.epoch)
        return self._keys

    def rotate(self, new_epoch: int) -> None:
        self.epoch = new_epoch

    def reader_keys(self, m: Manifest) -> JobKeys:
        ep = int(m.doc["key_epoch"])
        wrapped = base64.b64decode(m.doc["wrapped_dek"])
        k = self._deks.get((ep, wrapped))
        if k is None:
            k = JobKeys.adopt(self.ks, self.job, ep, wrapped)
            self._deks[(ep, wrapped)] = k
        return k

    def make_doc(
        self, stream: str, name: str, seq: int, prev: str | None, step: int, shards: list[dict]
    ) -> dict:
        k = self.writer_keys()
        return {
            "v": 1,
            "job": self.job,
            "stream": stream,
            "name": name,
            "seq": seq,
            "prev_manifest_hash": prev,
            "step": step,
            "created_at": time.time(),
            "key_epoch": k.epoch,
            "wrapped_dek": base64.b64encode(k.wrapped_dek).decode(),
            "shards": shards,
        }

    def sign(self, doc: dict) -> Manifest:
        if self.sk is None:
            raise ValueError("read-only envelope (no signing key)")
        return sign_manifest(self.sk, doc)

    def verify(self, body: bytes) -> Manifest:
        return verify_manifest(self.pk, body, self.job)


def new_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()
