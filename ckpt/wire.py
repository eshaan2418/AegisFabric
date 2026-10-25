"""Client for the `aegis-ckpt` length-prefixed binary protocol (see ckpt/src/server.rs)."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Sequence

from ckpt.errors import IntegrityError, InvalidRequest, NotFoundError, ServerError

OP_PUT_CHUNK = 1
OP_HAS_CHUNKS = 2
OP_COMMIT = 3
OP_LATEST = 4
OP_GET_MANIFEST = 5
OP_GET_CHUNK = 6
OP_GC = 7
OP_STATS = 8
OP_PUT_CHUNKS = 9
OP_VERIFY = 10
OP_LIST = 11

_ERR = {1: ServerError, 2: IntegrityError, 3: NotFoundError, 4: InvalidRequest}


def parse_addr(addr: str | tuple[str, int]) -> tuple[str, int]:
    if isinstance(addr, tuple):
        return addr
    host, port = addr.rsplit(":", 1)
    return host, int(port)


class Conn:
    def __init__(self, addr: str | tuple[str, int], timeout: float = 120.0):
        self.addr = parse_addr(addr)
        self.sock = socket.create_connection(self.addr, timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ServerError("connection closed by server")
            got += k
        return bytes(buf)

    def call(self, op: int, *parts: bytes | memoryview) -> bytes:
        n = 1 + sum(len(p) for p in parts)
        self.sock.sendall(struct.pack("<IB", n, op))
        for p in parts:
            self.sock.sendall(p)
        (ln,) = struct.unpack("<I", self._recv_exact(4))
        body = self._recv_exact(ln)
        status, payload = body[0], body[1:]
        if status != 0:
            raise _ERR.get(status, ServerError)(payload.decode(errors="replace"))
        return payload

    # --- typed ops ---------------------------------------------------------------------------
    def put_chunks(self, chunks: Sequence[tuple[bytes, bytes | memoryview]]) -> list[bool]:
        parts: list[bytes | memoryview] = [struct.pack("<I", len(chunks))]
        for cid, data in chunks:
            parts.append(cid + struct.pack("<I", len(data)))
            parts.append(data)
        r = self.call(OP_PUT_CHUNKS, b"".join(parts))
        return [b == 1 for b in r]

    def has_chunks(self, ids: Sequence[bytes]) -> list[bool]:
        r = self.call(OP_HAS_CHUNKS, struct.pack("<I", len(ids)) + b"".join(ids))
        return [b == 1 for b in r]

    def commit(self, seq: int, ids: Sequence[bytes], blob: bytes) -> None:
        self.call(OP_COMMIT, struct.pack("<QI", seq, len(ids)) + b"".join(ids) + blob)

    def latest(self) -> int | None:
        try:
            return struct.unpack("<Q", self.call(OP_LATEST))[0]
        except NotFoundError:
            return None

    def list(self) -> list[int]:
        r = self.call(OP_LIST)
        (n,) = struct.unpack("<I", r[:4])
        return list(struct.unpack(f"<{n}Q", r[4 : 4 + 8 * n]))

    def get_manifest(self, seq: int = 0) -> tuple[int, list[bytes], bytes]:
        r = self.call(OP_GET_MANIFEST, struct.pack("<Q", seq))
        s, n = struct.unpack("<QI", r[:12])
        ids = [r[12 + 32 * i : 44 + 32 * i] for i in range(n)]
        return s, ids, r[12 + 32 * n :]

    def get_chunk(self, cid: bytes) -> bytes:
        return self.call(OP_GET_CHUNK, cid)

    def gc(self, keep: int) -> dict:
        return json.loads(self.call(OP_GC, struct.pack("<I", keep)))

    def stats(self) -> dict:
        return json.loads(self.call(OP_STATS))

    def verify(self, seq: int = 0) -> dict:
        return json.loads(self.call(OP_VERIFY, struct.pack("<Q", seq)))
