"""CheckpointClient: snapshot → background flush to an `aegis-ckpt` server, and verified restore.

Chunking: fixed-size chunks per tensor (default 1 MiB). Tensors in a training state have a fixed byte
layout between steps (no insertions/shifts), so content-defined chunking (FastCDC) buys nothing for
dedup here, and fixed-size, tensor-aligned chunks are cheaper. Unchanged tensors (frozen weights) and
unchanged regions of partially updated tensors (e.g. embedding rows that received no gradient) dedup
exactly.

The storage server address is a constructor parameter. The M3 controller picks the storage site, and
routing is per destination prefix, so the address alone decides which path checkpoint traffic takes.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import blake3
import torch

from ckpt.crypto import CkptCrypto
from ckpt.errors import (
    CkptError,
    IntegrityError,
    InvalidRequest,
    NotFoundError,
    RollbackError,
    ServerError,
    SignatureError,
)
from ckpt.wire import Conn

MANIFEST_MAGIC = b"AGM1"
SIG_LEN = 64


@dataclass
class SaveStats:
    seq: int = 0
    step: int = 0
    tensors: int = 0
    chunks: int = 0
    logical_bytes: int = 0
    unique_chunks: int = 0
    dedup_hits: int = 0  # chunks skipped because the server already had them
    uploaded_bytes: int = 0  # bytes actually sent in PUT payloads (ciphertext if encrypted)
    stall_s: float = 0.0  # time the training loop was blocked in snapshot()
    flush_s: float = 0.0  # background time from snapshot end to durable commit ack
    hash_s: float = 0.0
    encrypt_s: float = 0.0
    upload_s: float = 0.0
    retries: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RestoreInfo:
    seq: int
    step: int
    chunks: int
    bytes: int
    fetch_s: float
    verify_s: float
    total_s: float
    manifest_hash: str
    extra: dict = field(default_factory=dict)


class SaveHandle:
    def __init__(self, stats: SaveStats):
        self.stats = stats
        self.done = threading.Event()
        self.exc: BaseException | None = None

    def wait(self, timeout: float | None = None) -> SaveStats:
        if not self.done.wait(timeout):
            raise TimeoutError("checkpoint flush did not finish")
        if self.exc is not None:
            raise self.exc
        return self.stats


def _tensor_bytes(t: torch.Tensor) -> memoryview:
    if t.numel() == 0:
        return memoryview(b"")
    return memoryview(t.reshape(-1).view(torch.uint8).numpy())


_DTYPES = {
    str(d): d
    for d in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ]
}


def manifest_hash(blob: bytes) -> str:
    return blake3.blake3(blob).hexdigest()


class CheckpointClient:
    def __init__(
        self,
        addr: str | tuple[str, int],
        crypto: CkptCrypto | None,
        job: str = "job",
        chunk_size: int = 1 << 20,
        batch_bytes: int = 16 << 20,
        connections: int = 2,
        timeout: float = 300.0,
        max_retries: int = 5,
        retry_backoff: float = 0.05,
    ):
        self.addr = addr
        self.crypto = crypto
        self.job = crypto.job if crypto else job
        self.chunk_size = chunk_size
        self.batch_bytes = batch_bytes
        self.nconn = max(1, connections)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._ctl = Conn(addr, timeout)
        self._pool: queue.Queue[Conn] = queue.Queue()
        for _ in range(self.nconn):
            self._pool.put(Conn(addr, timeout))
        self._exec = ThreadPoolExecutor(max_workers=self.nconn, thread_name_prefix="ckpt-up")
        self._flusher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-flush")
        self._inflight: SaveHandle | None = None
        self._lock = threading.Lock()
        # anti-rollback state: highest seq this client committed and the hash of that manifest
        self.hwm = 0
        self.anchor: tuple[int, str] | None = None

    def close(self) -> None:
        if self._inflight is not None:
            try:
                self._inflight.wait(self.timeout)
            except Exception:
                pass
        self._exec.shutdown(wait=True)
        self._flusher.shutdown(wait=True)
        self._ctl.close()
        while not self._pool.empty():
            self._pool.get().close()

    def __enter__(self) -> CheckpointClient:
        return self

    def __exit__(self, *a) -> None:
        self.close()

    # --- chunk ids / encoding -------------------------------------------------------------------
    def _cid(self, data: memoryview) -> bytes:
        if self.crypto is not None:
            return self.crypto.chunk_id(data)
        return blake3.blake3(data).digest()

    def _encode(self, cid: bytes, data: memoryview) -> bytes | memoryview:
        return self.crypto.encrypt_chunk(cid, data) if self.crypto is not None else data

    def _decode(self, cid: bytes, blob: bytes, wrapped: dict[int, bytes]) -> bytes:
        if self.crypto is not None:
            return self.crypto.decrypt_chunk(cid, blob, wrapped)
        if blake3.blake3(blob).digest() != cid:
            raise IntegrityError("chunk id does not match content")
        return blob

    # --- save -----------------------------------------------------------------------------------
    def snapshot(
        self, state: dict[str, torch.Tensor], step: int = 0, extra: dict | None = None
    ) -> SaveHandle:
        """Copy `state` to host memory (the only part that blocks training) and flush in background.

        If a previous flush is still running, snapshot first waits for it. That wait is counted in
        stall_s, because it is real back-pressure on the training loop."""
        t0 = time.perf_counter()
        if self._inflight is not None:
            self._inflight.wait(self.timeout)
        snap = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in state.items()}
        stats = SaveStats(step=step, tensors=len(snap))
        stats.stall_s = time.perf_counter() - t0
        h = SaveHandle(stats)
        self._inflight = h
        self._flusher.submit(self._flush, snap, step, extra or {}, h)
        return h

    def save(self, state: dict[str, torch.Tensor], step: int = 0, extra: dict | None = None) -> SaveStats:
        return self.snapshot(state, step, extra).wait(self.timeout)

    def wait(self) -> SaveStats | None:
        return self._inflight.wait(self.timeout) if self._inflight is not None else None

    def _upload(self, batch: list[tuple[bytes, memoryview]], stats: SaveStats) -> None:
        t0 = time.perf_counter()
        enc = [(cid, self._encode(cid, data)) for cid, data in batch]
        t1 = time.perf_counter()
        # Liveness (TLA+ finding): a PUT whose ack is lost (timeout, reset) is resent on a fresh
        # connection before COMMIT; PUT is idempotent because chunks are content-addressed.
        c = self._pool.get()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    c.put_chunks(enc)
                    break
                except (OSError, ServerError):
                    if attempt == self.max_retries:
                        raise
                    with self._lock:
                        stats.retries += 1
                    c.close()
                    time.sleep(self.retry_backoff * (2**attempt))
                    c = self._reconnect()
        finally:
            self._pool.put(c)
        with self._lock:
            stats.encrypt_s += t1 - t0
            stats.upload_s += time.perf_counter() - t1
            stats.uploaded_bytes += sum(len(e) for _, e in enc)

    def _flush(self, snap: dict[str, torch.Tensor], step: int, extra: dict, h: SaveHandle) -> None:
        t0 = time.perf_counter()
        st = h.stats
        try:
            shards = []
            uniq: dict[bytes, memoryview] = {}
            for name in sorted(snap):
                t = snap[name]
                mv = _tensor_bytes(t)
                ids = []
                for off in range(0, len(mv), self.chunk_size):
                    piece = mv[off : off + self.chunk_size]
                    cid = self._cid(piece)
                    ids.append(cid)
                    uniq.setdefault(cid, piece)
                st.chunks += len(ids)
                st.logical_bytes += len(mv)
                shards.append(
                    {
                        "name": name,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "nbytes": len(mv),
                        "chunks": [i.hex() for i in ids],
                    }
                )
            st.unique_chunks = len(uniq)
            st.hash_s = time.perf_counter() - t0
            ordered = list(uniq)
            self._send_missing(ordered, uniq, st, check_server=True)
            self._commit(shards, ordered, step, extra, st, uniq)
            st.flush_s = time.perf_counter() - t0
        except BaseException as e:  # surfaced to the caller via wait()
            st.error = repr(e)
            h.exc = e
        finally:
            h.done.set()

    def _send_missing(
        self, ordered: list[bytes], uniq: dict[bytes, memoryview], st: SaveStats, check_server: bool
    ) -> None:
        missing = []
        for i in range(0, len(ordered), 4096):
            part = ordered[i : i + 4096]
            have = self._ctl.has_chunks(part) if check_server else [False] * len(part)
            for cid, hv in zip(part, have, strict=True):
                if hv:
                    st.dedup_hits += 1
                else:
                    missing.append(cid)
        futs = []
        batch: list[tuple[bytes, memoryview]] = []
        size = 0
        for cid in missing:
            batch.append((cid, uniq[cid]))
            size += len(uniq[cid])
            if size >= self.batch_bytes:
                futs.append(self._exec.submit(self._upload, batch, st))
                batch, size = [], 0
        if batch:
            futs.append(self._exec.submit(self._upload, batch, st))
        for f in futs:
            f.result()

    def _commit(self, shards, ordered, step, extra, st: SaveStats, uniq) -> None:
        # Several writers (e.g. the ranks of one job, each saving its slice) may share one server and
        # its single manifest sequence. If another writer takes `seq` first, the server rejects ours as
        # non-monotonic; re-derive seq/prev from the new head and re-sign (chunks are already durable).
        for _race in range(64):
            try:
                return self._commit_once(shards, ordered, step, extra, st, uniq)
            except InvalidRequest:
                latest = self._ctl.latest() or 0
                if latest < max(self.hwm, st.seq):
                    raise
                st.retries += 1
        raise CkptError("commit lost the sequence race 64 times")

    def _commit_once(self, shards, ordered, step, extra, st: SaveStats, uniq) -> None:
        latest = self._ctl.latest() or 0
        seq = max(latest, self.hwm) + 1
        st.seq = seq
        prev = ""
        if latest:
            _, _, pblob = self._ctl.get_manifest(latest)
            prev = manifest_hash(pblob)
        body = {
            "job": self.job,
            "seq": seq,
            "prev_manifest_hash": prev,
            "step": step,
            "created_at": time.time(),
            "shards": shards,
            "extra": extra,
        }
        if self.crypto is not None:
            body["key_epoch"] = self.crypto.key_epoch
            # Deduplicated chunks may have been encrypted under any earlier key epoch, so every
            # wrapped DEK the job knows is carried (each is ~60 bytes; only the KMS can unwrap them).
            body["wrapped_deks"] = {str(e): w.hex() for e, w in self.crypto.wrapped_deks().items()}
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        sig = self.crypto.sign(raw) if self.crypto is not None else bytes(SIG_LEN)
        blob = MANIFEST_MAGIC + sig + raw
        for attempt in range(self.max_retries + 1):
            try:
                self._ctl.commit(seq, ordered, blob)
                break
            except NotFoundError:
                # A GC between HAS_CHUNKS and COMMIT collected a chunk we deduplicated against:
                # re-upload everything this manifest references, then retry.
                st.retries += 1
                self._send_missing(ordered, uniq, st, check_server=False)
            except (OSError, ServerError, InvalidRequest) as e:
                # Lost COMMIT ack: if the server already holds exactly our manifest at `seq`, done.
                if attempt == self.max_retries:
                    raise
                st.retries += 1
                if not isinstance(e, InvalidRequest):
                    self._ctl.close()
                    time.sleep(self.retry_backoff * (2**attempt))
                    self._ctl = self._reconnect()
                if self._committed_as(seq, blob):
                    break
                if isinstance(e, InvalidRequest):
                    raise
        else:
            raise CkptError(f"commit of seq {seq} failed after {self.max_retries} retries")
        st.seq = seq
        self.hwm = seq
        self.anchor = (seq, manifest_hash(blob))

    def _reconnect(self) -> Conn:
        return Conn(self.addr, self.timeout)

    def _committed_as(self, seq: int, blob: bytes) -> bool:
        try:
            sseq, _, sblob = self._ctl.get_manifest(seq)
        except (NotFoundError, OSError, ServerError):
            return False
        return sseq == seq and sblob == blob

    # --- restore --------------------------------------------------------------------------------
    def _parse_manifest(self, server_seq: int, ids: list[bytes], blob: bytes) -> dict:
        if blob[:4] != MANIFEST_MAGIC or len(blob) < 4 + SIG_LEN:
            raise SignatureError("bad manifest framing")
        sig, raw = blob[4 : 4 + SIG_LEN], blob[4 + SIG_LEN :]
        if self.crypto is not None:
            self.crypto.verify(sig, raw)
        try:
            m = json.loads(raw)
        except ValueError as e:
            raise IntegrityError("manifest JSON invalid") from e
        if m.get("seq") != server_seq:
            raise SignatureError(f"server header seq {server_seq} != signed seq {m.get('seq')} (replay?)")
        if m.get("job") != self.job:
            raise SignatureError("manifest belongs to a different job")
        listed = {bytes.fromhex(c) for s in m["shards"] for c in s["chunks"]}
        if listed != set(ids):
            raise SignatureError("server chunk list does not match signed manifest")
        return m

    def _check_chain(self, target_seq: int, target_hash: str, anchor: tuple[int, str]) -> None:
        aseq, ahash = anchor
        if target_seq == aseq:
            if target_hash != ahash:
                raise RollbackError("manifest at anchor seq differs from the trusted one")
            return
        if target_seq < aseq:
            raise RollbackError(f"seq {target_seq} older than trusted anchor {aseq}")
        prev_hash = ahash
        for s in range(aseq + 1, target_seq + 1):
            try:
                sseq, ids, blob = self._ctl.get_manifest(s)
            except NotFoundError as e:
                raise RollbackError(f"hash chain broken: manifest {s} missing") from e
            m = self._parse_manifest(sseq, ids, blob)
            if m["prev_manifest_hash"] != prev_hash:
                raise RollbackError(f"hash chain broken at seq {s}")
            prev_hash = manifest_hash(blob)

    def restore(
        self,
        seq: int | None = None,
        trusted_hwm: int | None = None,
        trusted_anchor: tuple[int, str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], RestoreInfo]:
        """Fetch and fully verify a checkpoint (default: latest).

        trusted_hwm: the lowest seq the caller will accept (anti-rollback; the controller keeps it).
        trusted_anchor: (seq, manifest_hash) known good; verifies the prev-hash chain from it.
        Raises IntegrityError (or a subclass) on any detected problem; never returns unverified data."""
        t0 = time.perf_counter()
        sseq, ids, blob = self._ctl.get_manifest(seq or 0)
        m = self._parse_manifest(sseq, ids, blob)
        if trusted_hwm is not None and sseq < trusted_hwm:
            raise RollbackError(f"latest manifest seq {sseq} < trusted high-water mark {trusted_hwm}")
        mh = manifest_hash(blob)
        if trusted_anchor is not None:
            self._check_chain(sseq, mh, trusted_anchor)
        wrapped = {int(k): bytes.fromhex(v) for k, v in m.get("wrapped_deks", {}).items()}
        if self.crypto is None and wrapped:
            raise CkptError("checkpoint is encrypted but client has no crypto config")

        def fetch(cid: bytes) -> tuple[bytes, bytes]:
            c = self._pool.get()
            try:
                return cid, c.get_chunk(cid)
            except NotFoundError as e:
                raise IntegrityError(f"chunk {cid.hex()[:16]} missing") from e
            finally:
                self._pool.put(c)

        uniq = list(dict.fromkeys(ids))
        t1 = time.perf_counter()
        raw = dict(self._exec.map(fetch, uniq))
        t2 = time.perf_counter()
        plain = {cid: self._decode(cid, raw[cid], wrapped) for cid in uniq}
        out: dict[str, torch.Tensor] = {}
        total = 0
        for s in m["shards"]:
            dt = _DTYPES[s["dtype"]]
            buf = bytearray(b"".join(plain[bytes.fromhex(c)] for c in s["chunks"]))
            if len(buf) != s["nbytes"]:
                raise IntegrityError(f"tensor {s['name']} size mismatch")
            total += len(buf)
            if len(buf) == 0:
                out[s["name"]] = torch.empty(s["shape"], dtype=dt)
            else:
                out[s["name"]] = torch.frombuffer(buf, dtype=dt).reshape(s["shape"])
        t3 = time.perf_counter()
        info = RestoreInfo(
            seq=sseq,
            step=m.get("step", 0),
            chunks=len(uniq),
            bytes=total,
            fetch_s=t2 - t1,
            verify_s=t3 - t2,
            total_s=t3 - t0,
            manifest_hash=mh,
            extra=m.get("extra", {}),
        )
        return out, info

    # --- admin ----------------------------------------------------------------------------------
    def latest(self) -> int | None:
        return self._ctl.latest()

    def list(self) -> list[int]:
        return self._ctl.list()

    def gc(self, keep: int) -> dict:
        return self._ctl.gc(keep)

    def stats(self) -> dict:
        return self._ctl.stats()

    def verify(self, seq: int = 0) -> dict:
        return self._ctl.verify(seq)


class EngineBackend:
    """Blob backend with the `netlab.blobstore.Client` interface: `put(key, bytes) -> seconds`,
    `get(key) -> bytes | None`. `train.checkpointing.SliceCheckpointer` uses it unchanged, and
    `train.trainer --ckpt-backend engine` constructs it as `EngineBackend(addr, job=..., rank=...)`.

    Each put is one committed checkpoint whose only shard is the blob (a uint8 tensor in fixed-size
    chunks), with `{"key": key}` in the (signed) manifest's `extra`. Successive puts of similar blobs
    (successive training checkpoints of one slice) upload only the chunks the server does not already
    hold. get(key) restores the newest manifest carrying that key, fully verified. With `crypto` the
    blob is encrypted and the manifest signed (P2). The seq recorded at put time is the lower bound
    enforced at get time (anti-rollback; pass a shared `trusted` dict to extend it beyond this object).
    """

    def __init__(
        self,
        addr: str,
        job: str = "job",
        rank: int = 0,
        crypto: CkptCrypto | None = None,
        chunk_size: int = 256 << 10,
        connections: int = 2,
        timeout: float = 300.0,
        trusted: dict[str, int] | None = None,
    ):
        self.rank = rank
        self.cli = CheckpointClient(
            addr, crypto, job=job, chunk_size=chunk_size, connections=connections, timeout=timeout
        )
        self.trusted = trusted if trusted is not None else {}
        self._index: dict[str, int] = {}  # key -> newest seq seen carrying it
        self._scanned: set[int] = set()
        self.last_put: SaveStats | None = None
        self.last_get: RestoreInfo | None = None
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes) -> float:
        t0 = time.perf_counter()
        if data:
            blob = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        else:
            blob = torch.empty(0, dtype=torch.uint8)
        with self._lock:
            st = self.cli.save({"blob": blob}, step=0, extra={"key": key})
            self.last_put = st
            self._index[key] = st.seq
            self.trusted[key] = max(self.trusted.get(key, 0), st.seq)
        return time.perf_counter() - t0

    def _find(self, key: str) -> int | None:
        for s in sorted(set(self.cli.list()) - self._scanned, reverse=True):
            try:
                _, _, blob = self.cli._ctl.get_manifest(s)
                k = json.loads(blob[4 + SIG_LEN :]).get("extra", {}).get("key")
            except (NotFoundError, ValueError):
                continue
            self._scanned.add(s)
            if k is not None and self._index.get(k, 0) < s:
                self._index[k] = s
        return self._index.get(key)

    def keys(self) -> set[str]:
        """Blob keys with at least one committed manifest (read from manifests without verifying them;
        get() verifies). Used by the job manager to find the newest complete checkpoint after a crash."""
        with self._lock:
            self._find("")
            return set(self._index)

    def get(self, key: str) -> bytes | None:
        with self._lock:
            seq = self._find(key)
            if seq is None:
                if key in self.trusted:
                    raise RollbackError(f"{key}: committed as seq {self.trusted[key]} but not on the server")
                return None
            out, info = self.cli.restore(seq=seq, trusted_hwm=self.trusted.get(key))
            if info.extra.get("key") != key:
                raise SignatureError(f"manifest {seq} is not for key {key!r}")
            self.last_get = info
            return out["blob"].numpy().tobytes()

    def close(self) -> None:
        self.cli.close()
