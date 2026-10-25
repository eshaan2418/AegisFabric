"""Python client for the aegis-ckpt engine (Rust server in this directory; protocol P1 + P2).

Drop-in replacement for `netlab.blobstore.Client`: `put(key, data) -> seconds` and
`get(key) -> bytes | None`, so `train.checkpointing.SliceCheckpointer` uses it unchanged. On top of the
blob interface it gives

* incremental checkpoints: data is split into fixed-size chunks, the server is asked which chunk ids it
  already holds (HAS), and only new chunks cross the network (PUT_BATCH, group-committed and fsync'd
  server-side), then a manifest is COMMITted (WAL append + fsync before the ack);
* end-to-end verification on `get`: every chunk is re-hashed (plaintext mode) or AES-GCM-opened and its
  keyed id recomputed (secure mode, `security.envelope`), the manifest signature / header binding is
  checked, and the manifest sequence number is checked against a trust store (anti-rollback);
* streams: every key maps to a stream (default: the key with its 8-digit step removed, i.e. one
  stream per checkpoint slice). Sequence numbers and the prev-hash chain are per stream.

Enabling it in the WAN harness: start `aegis-ckpt serve --root DIR --addr 0.0.0.0:PORT` at each
storage site instead of `python -m netlab.blobstore`, and pass `--ckpt-backend engine` to
`train.trainer` (which constructs `EngineBackend(addr, job=..., rank=...)`). Set `AEGIS_CKPT_TRUST` to a
shared JSON path for cross-process rollback protection, and `AEGIS_CKPT_SECURE=1` (+ key material, see
`EngineBackend.__init__`) for encryption.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import blake3

from security import envelope as env

OP_PUT_BATCH, OP_HAS, OP_COMMIT, OP_GET_MANIFEST, OP_GET_CHUNKS, OP_LATEST = 1, 2, 3, 4, 5, 6
OP_GC, OP_STATS, OP_VERIFY, OP_GET_MANIFEST_SEQ, OP_LIST, OP_RECOVERY = 7, 8, 9, 10, 11, 12
ST_OK, ST_NOT_FOUND, ST_ERROR, ST_MISSING = 0, 1, 2, 3

CRATE = Path(__file__).resolve().parent
DEFAULT_CHUNK = 256 * 1024


class EngineError(Exception):
    """Server-reported error (includes detected corruption)."""


class MissingChunks(EngineError):
    def __init__(self, ids: list[bytes]):
        super().__init__(f"{len(ids)} chunks not durable on the server")
        self.ids = ids


# ------------------------------------------------------------------------------------ wire format


def encode_manifest(name: str, stream: str, mseq: int, ids: list[bytes], body: bytes) -> bytes:
    n, s = name.encode(), stream.encode()
    out = [b"AGM1", struct.pack("<H", len(n)), n, struct.pack("<H", len(s)), s]
    out += [struct.pack("<QI", mseq, len(ids)), *ids, struct.pack("<I", len(body)), body]
    return b"".join(out)


def parse_manifest(b: bytes) -> tuple[str, str, int, list[bytes], bytes]:
    try:
        if b[:4] != b"AGM1":
            raise ValueError("magic")
        p = 4
        (n,) = struct.unpack_from("<H", b, p)
        name = b[p + 2 : p + 2 + n].decode()
        p += 2 + n
        (n,) = struct.unpack_from("<H", b, p)
        stream = b[p + 2 : p + 2 + n].decode()
        p += 2 + n
        mseq, k = struct.unpack_from("<QI", b, p)
        p += 12
        ids = [b[p + 32 * i : p + 32 * (i + 1)] for i in range(k)]
        p += 32 * k
        (bl,) = struct.unpack_from("<I", b, p)
        body = b[p + 4 : p + 4 + bl]
        if p + 4 + bl != len(b) or len(body) != bl:
            raise ValueError("length")
        return name, stream, mseq, ids, body
    except (ValueError, struct.error, UnicodeDecodeError) as e:
        raise env.BadManifest(f"unparseable manifest header: {e}") from e


class Conn:
    """One TCP connection speaking the framed protocol (thread-safe via a lock)."""

    def __init__(self, addr: str, timeout: float = 120.0):
        host, port = addr.rsplit(":", 1)
        self.addr = (host, int(port))
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sock: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self.sock is None:
            s = socket.create_connection(self.addr, timeout=self.timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            self.rf = s.makefile("rb", buffering=1 << 20)
        return self.sock

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def call(self, op: int, body: bytes | list[bytes] = b"") -> tuple[int, bytes]:
        parts = body if isinstance(body, list) else [body]
        n = sum(len(p) for p in parts)
        with self.lock:
            try:
                s = self._connect()
                s.sendall(struct.pack("<BI", op, n))
                for p in parts:
                    s.sendall(p)
                hdr = self.rf.read(5)
                if len(hdr) != 5:
                    raise ConnectionError("server closed the connection")
                st, ln = struct.unpack("<BI", hdr)
                out = self.rf.read(ln)
                if len(out) != ln:
                    raise ConnectionError("short reply")
                return st, out
            except OSError:
                self.close()
                raise


class EngineClient:
    """Low-level API, one method per server op."""

    def __init__(self, addr: str, timeout: float = 120.0):
        self.conn = Conn(addr, timeout)

    def _ok(self, st: int, b: bytes) -> bytes:
        if st == ST_OK:
            return b
        if st == ST_MISSING:
            raise MissingChunks([bytes.fromhex(h) for h in json.loads(b)])
        raise EngineError(b.decode(errors="replace"))

    def put_batch(self, chunks: list[tuple[bytes, bytes]]) -> dict:
        parts = [struct.pack("<I", len(chunks))]
        for cid, data in chunks:
            parts += [cid, struct.pack("<I", len(data)), data]
        return json.loads(self._ok(*self.conn.call(OP_PUT_BATCH, parts)))

    def has(self, ids: list[bytes]) -> list[bool]:
        return [x == 1 for x in self._ok(*self.conn.call(OP_HAS, b"".join(ids)))]

    def commit(self, manifest: bytes) -> dict:
        return json.loads(self._ok(*self.conn.call(OP_COMMIT, manifest)))

    def _maybe(self, op: int, body: bytes) -> bytes | None:
        st, b = self.conn.call(op, body)
        return None if st == ST_NOT_FOUND else self._ok(st, b)

    def get_manifest(self, name: str) -> bytes | None:
        return self._maybe(OP_GET_MANIFEST, name.encode())

    def latest(self, stream: str = "") -> bytes | None:
        return self._maybe(OP_LATEST, stream.encode())

    def get_manifest_seq(self, stream: str, mseq: int) -> bytes | None:
        s = stream.encode()
        return self._maybe(OP_GET_MANIFEST_SEQ, struct.pack("<H", len(s)) + s + struct.pack("<Q", mseq))

    def get_chunks(self, ids: list[bytes]) -> list[bytes]:
        b = memoryview(self._ok(*self.conn.call(OP_GET_CHUNKS, b"".join(ids))))
        out, p = [], 0
        for _ in ids:
            (n,) = struct.unpack_from("<I", b, p)
            out.append(bytes(b[p + 4 : p + 4 + n]))
            p += 4 + n
        return out

    def gc(self, keep: int = 2, min_dead: float = 0.3) -> dict:
        return json.loads(
            self._ok(*self.conn.call(OP_GC, json.dumps({"keep": keep, "min_dead": min_dead}).encode()))
        )

    def stats(self) -> dict:
        return json.loads(self._ok(*self.conn.call(OP_STATS)))

    def verify(self) -> dict:
        return json.loads(self._ok(*self.conn.call(OP_VERIFY)))

    def list(self) -> list[dict]:
        return json.loads(self._ok(*self.conn.call(OP_LIST)))

    def recovery(self) -> dict:
        return json.loads(self._ok(*self.conn.call(OP_RECOVERY)))

    def close(self) -> None:
        self.conn.close()


# ------------------------------------------------------------------------------------ the backend

_STEP_RE = re.compile(r"^(.*)_(\d{8})_(.*)$")


def default_stream(key: str) -> tuple[str, int]:
    """`ckpt_<job>_<step:08d>_slice<r>of<n>` -> (`ckpt_<job>_slice<r>of<n>`, step); else (key, -1)."""
    m = _STEP_RE.match(key)
    if not m:
        return key, -1
    return f"{m.group(1)}_{m.group(3)}", int(m.group(2))


def fixed_chunks(data: bytes | memoryview, size: int) -> list[memoryview]:
    mv = memoryview(data)
    if len(mv) == 0:
        return [mv]
    return [mv[i : i + size] for i in range(0, len(mv), size)]


class EngineBackend:
    """put/get blob backend over the engine. See the module docstring."""

    def __init__(
        self,
        addr: str,
        job: str = "job",
        rank: int = 0,
        *,
        envelope: env.Envelope | None = None,
        trust: env.TrustStore | None = None,
        chunk_size: int = DEFAULT_CHUNK,
        batch_bytes: int = 8 << 20,
        timeout: float = 120.0,
        stream_fn=default_stream,
    ):
        self.addr, self.job, self.rank = addr, job, rank
        self.cli = EngineClient(addr, timeout)
        self.envelope = envelope if envelope is not None else _env_from_environment(job)
        tp = os.environ.get("AEGIS_CKPT_TRUST")
        self.trust = trust if trust is not None else env.TrustStore(Path(tp) if tp else None)
        self.chunk_size, self.batch_bytes, self.stream_fn = chunk_size, batch_bytes, stream_fn
        self.lock = threading.Lock()
        self.last_put: dict = {}
        self.last_get: dict = {}

    # -- manifests --------------------------------------------------------------------------
    def _open(self, raw: bytes) -> tuple[env.Manifest, str, str, int, list[bytes]]:
        name, stream, mseq, ids, body = parse_manifest(raw)
        if self.envelope is not None:
            m = self.envelope.verify(body)
        else:
            m = env.parse_unsigned(body)
            if m.doc.get("job") != self.job:
                raise env.BadManifest(f"manifest is for job {m.doc.get('job')!r}")
        d = m.doc
        if (d.get("name"), d.get("stream"), d.get("seq")) != (name, stream, mseq) or m.chunk_ids != ids:
            raise env.BadManifest("server-visible manifest header does not match the manifest content")
        return m, name, stream, mseq, ids

    def _fetch_seq(self, stream: str):
        def f(seq: int) -> env.Manifest | None:
            raw = self.cli.get_manifest_seq(stream, seq)
            return None if raw is None else self._open(raw)[0]

        return f

    def _next_seq(self, stream: str) -> tuple[int, str | None]:
        hwm, h = self.trust.hwm(stream)
        raw = self.cli.latest(stream)
        if raw is not None:
            m = self._open(raw)[0]
            self.trust.check(stream, m, m.doc["name"], self._fetch_seq(stream), latest=True)
            if m.seq >= hwm:
                if self.envelope is not None and m.doc.get("key_epoch") == self.envelope.epoch:
                    self.envelope.writer_keys((m.doc["key_epoch"], _b64(m.doc["wrapped_dek"])))
                return m.seq + 1, m.hash
        return hwm + 1, h

    # -- put --------------------------------------------------------------------------------
    def put(self, key: str, data: bytes, retries: int = 3) -> float:
        t0 = time.perf_counter()
        stream, step = self.stream_fn(key)
        with self.lock:
            for attempt in range(retries):
                try:
                    self._put_once(key, stream, step, data)
                    break
                except (OSError, ConnectionError):
                    self.cli.close()
                    if attempt == retries - 1:
                        raise
                    time.sleep(0.5)
        self.last_put["put_s"] = time.perf_counter() - t0
        return self.last_put["put_s"]

    def _put_once(self, key: str, stream: str, step: int, data: bytes) -> None:
        t0 = time.perf_counter()
        seq, prev = self._next_seq(stream)
        pieces = fixed_chunks(data, self.chunk_size)
        if self.envelope is not None:
            keys = self.envelope.writer_keys()
            ids, payload = [], {}
            for p in pieces:
                cid = env.chunk_id(keys.mac_key, p)
                ids.append(cid)
                payload[cid] = p
        else:
            ids = [blake3.blake3(p).digest() for p in pieces]
            payload = dict(zip(ids, pieces, strict=True))
        t_hash = time.perf_counter()
        uniq = list(dict.fromkeys(ids))
        have = self.cli.has(uniq)
        missing = [c for c, h in zip(uniq, have, strict=True) if not h]
        sent = self._send(missing, payload)
        t_send = time.perf_counter()
        shards = [{"name": key, "chunks": [c.hex() for c in ids], "len": len(data)}]
        if self.envelope is not None:
            m = self.envelope.sign(self.envelope.make_doc(stream, key, seq, prev, step, shards))
        else:
            doc = {"v": 1, "mode": "plain", "job": self.job, "stream": stream, "name": key, "seq": seq}
            doc |= {"prev_manifest_hash": prev, "step": step, "created_at": time.time(), "shards": shards}
            m = env.unsigned_manifest(doc)
        raw = encode_manifest(key, stream, seq, ids, m.body)
        resent = 0
        for _ in range(4):
            try:
                ack = self.cli.commit(raw)
                break
            except MissingChunks as e:  # chunk swept by a concurrent GC: resend, then retry (P1 liveness)
                resent += self._send(e.ids, payload)
        else:
            raise EngineError("commit kept failing with missing chunks")
        self.trust.record(stream, seq, m.hash, key)
        self.last_put = {
            "key": key,
            "seq": seq,
            "bytes": len(data),
            "chunks": len(ids),
            "unique_chunks": len(uniq),
            "new_chunks": len(missing),
            "sent_bytes": sent,
            "resent_bytes": resent,
            "hash_s": t_hash - t0,
            "send_s": t_send - t_hash,
            "commit_s": time.perf_counter() - t_send,
            "wal_seq": ack["wal_seq"],
            "wal_end": ack["wal_end"],
        }

    def _send(self, ids: list[bytes], payload: dict) -> int:
        batch, nb, sent = [], 0, 0
        keys = self.envelope.writer_keys() if self.envelope is not None else None
        for cid in ids:
            p = payload[cid]
            if keys is not None:
                blob = env.encrypt_chunk(keys, cid, bytes(p))
            else:
                blob = bytes(p)
            batch.append((cid, blob))
            nb += len(blob)
            if nb >= self.batch_bytes:
                self.cli.put_batch(batch)
                sent += nb
                batch, nb = [], 0
        if batch:
            self.cli.put_batch(batch)
            sent += nb
        return sent

    # -- get --------------------------------------------------------------------------------
    def get(self, key: str, retries: int = 3) -> bytes | None:
        for attempt in range(retries):
            try:
                with self.lock:
                    return self._get_once(key)
            except (OSError, ConnectionError):
                self.cli.close()
                if attempt == retries - 1:
                    raise
                time.sleep(0.5)
        return None

    def _get_once(self, key: str, latest: bool = False, raw: bytes | None = None) -> bytes | None:
        t0 = time.perf_counter()
        if raw is None:
            raw = self.cli.get_manifest(key)
        if raw is None:
            return None
        m, name, stream, _seq, ids = self._open(raw)
        if name != key and not latest:
            raise env.BadManifest(f"asked for {key!r}, server returned manifest of {name!r}")
        how = self.trust.check(stream, m, name, self._fetch_seq(stream), latest=latest)
        if how == "tofu" or how.startswith("chain"):
            self.trust.record(stream, m.seq, m.hash, name)
        t_man = time.perf_counter()
        keys = self.envelope.reader_keys(m) if self.envelope is not None else None
        uniq = list(dict.fromkeys(ids))
        plain: dict[bytes, bytes] = {}
        batch, nb = [], 0
        for cid in uniq:
            batch.append(cid)
            nb += self.chunk_size
            if nb >= self.batch_bytes:
                self._fetch_into(batch, keys, plain)
                batch, nb = [], 0
        if batch:
            self._fetch_into(batch, keys, plain)
        out = b"".join(plain[c] for c in ids)
        want = sum(s["len"] for s in m.doc["shards"])
        if len(out) != want:
            raise env.BadManifest(f"{key}: reassembled {len(out)} bytes, manifest says {want}")
        self.last_get = {
            "key": key,
            "seq": m.seq,
            "bytes": len(out),
            "chunks": len(ids),
            "manifest_s": t_man - t0,
            "total_s": time.perf_counter() - t0,
            "trust": how,
        }
        return out

    def _fetch_into(self, batch: list[bytes], keys, plain: dict) -> None:
        for cid, blob in zip(batch, self.cli.get_chunks(batch), strict=True):
            if keys is not None:
                plain[cid] = env.open_chunk(keys.dek, keys.mac_key, cid, blob)
            else:
                if blake3.blake3(blob).digest() != cid:
                    raise env.TamperedChunk(f"chunk {cid.hex()[:16]}: content does not match its id")
                plain[cid] = blob

    def restore_latest(self, stream: str) -> tuple[str, bytes] | None:
        """Newest committed checkpoint of a stream, with the high-water-mark check."""
        with self.lock:
            raw = self.cli.latest(stream)
            if raw is None:
                hwm, _ = self.trust.hwm(stream)
                if hwm:
                    raise env.Rollback(f"stream {stream}: server has no checkpoint but trusted hwm is {hwm}")
                return None
            name = parse_manifest(raw)[0]
            return name, self._get_once(name, latest=True, raw=raw)

    def close(self) -> None:
        self.cli.close()


def _b64(s: str) -> bytes:
    return base64.b64decode(s)


def _env_from_environment(job: str) -> env.Envelope | None:
    """AEGIS_CKPT_SECURE=1 with AEGIS_CKPT_KEK (hex, 32 B) and AEGIS_CKPT_SIGNING_KEY (hex Ed25519 seed)."""
    if os.environ.get("AEGIS_CKPT_SECURE") != "1":
        return None
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    kek = bytes.fromhex(os.environ["AEGIS_CKPT_KEK"])
    sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(os.environ["AEGIS_CKPT_SIGNING_KEY"]))
    return env.Envelope(
        job, env.KeyService(kek), sk, key_epoch=int(os.environ.get("AEGIS_CKPT_KEY_EPOCH", "1"))
    )


# ---------------------------------------------------------------------------------- server control


def binary() -> Path:
    return CRATE / "target" / "release" / "aegis-ckpt"


def ensure_built() -> Path:
    b = binary()
    if not b.exists():
        subprocess.run(["cargo", "build", "--release", "-q", "-j", "4"], cwd=CRATE, check=True)
    return b


def start_server(
    root: Path,
    io: str = "auto",
    addr: str = "127.0.0.1:0",
    seg_mb: int = 64,
    threads: int = 4,
    env_extra=None,
) -> tuple[subprocess.Popen, str, dict]:
    """Start `aegis-ckpt serve`; returns (process, "host:port", ready info incl. the recovery report)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ready = root.parent / f".{root.name}.ready.{os.getpid()}.{time.monotonic_ns()}"
    cmd = [str(ensure_built()), "serve", "--root", str(root), "--addr", addr, "--io", io]
    cmd += ["--seg-mb", str(seg_mb), "--threads", str(threads), "--ready-file", str(ready)]
    p = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env={**os.environ, **(env_extra or {})}
    )
    t0 = time.monotonic()
    while not ready.exists():
        if p.poll() is not None:
            raise EngineError(f"server exited: {p.stderr.read().decode(errors='replace')[-2000:]}")
        if time.monotonic() - t0 > 60:
            p.kill()
            raise EngineError("server did not become ready")
        time.sleep(0.005)
    info = json.loads(ready.read_text())
    ready.unlink()
    host, port = info["addr"].rsplit(":", 1)
    if host in ("0.0.0.0", "[::]"):
        host = "127.0.0.1"
    return p, f"{host}:{port}", info
