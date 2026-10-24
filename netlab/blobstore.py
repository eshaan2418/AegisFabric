"""Minimal networked blob store (PUT/GET of whole objects over TCP, written + fsync'd to disk).

Used as the checkpoint transport for the WAN experiments (each rank PUTs its checkpoint slice to the
storage host at site s; restores GET every slice). It is deliberately simple: no dedup, no encryption,
no crash consistency beyond write-to-temp + fsync + rename. The real engine (ckpt/, M4) replaces it.

Wire format: one JSON header line, then `len` raw bytes for PUT requests / GET replies.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        root: Path = self.server.root  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            req = json.loads(line)
            key = req["key"].replace("/", "_")
            if req["op"] == "put":
                n = req["len"]
                tmp = root / f".{key}.tmp"
                with open(tmp, "wb") as f:
                    left = n
                    while left:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            return
                        f.write(chunk)
                        left -= len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, root / key)
                self.wfile.write(b'{"ok": true}\n')
            elif req["op"] == "get":
                p = root / key
                if not p.exists():
                    self.wfile.write(json.dumps({"ok": False, "len": 0}).encode() + b"\n")
                    continue
                data = p.read_bytes()
                self.wfile.write(json.dumps({"ok": True, "len": len(data)}).encode() + b"\n")
                self.wfile.write(data)
            elif req["op"] == "list":
                keys = sorted(x.name for x in root.iterdir() if not x.name.startswith("."))
                body = json.dumps(keys).encode()
                self.wfile.write(json.dumps({"ok": True, "len": len(body)}).encode() + b"\n" + body)
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(root: Path, host: str, port: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    srv = Server((host, port), Handler)
    srv.root = root  # type: ignore[attr-defined]
    srv.serve_forever()


class Client:
    def __init__(self, addr: str, timeout: float = 120.0):
        host, port = addr.rsplit(":", 1)
        self.addr = (host, int(port))
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sock: socket.socket | None = None

    def _conn(self) -> socket.socket:
        if self.sock is None:
            self.sock = socket.create_connection(self.addr, timeout=self.timeout)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.rf = self.sock.makefile("rb")
        return self.sock

    def _reset(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def put(self, key: str, data: bytes, retries: int = 3) -> float:
        for attempt in range(retries):
            try:
                with self.lock:
                    t0 = time.perf_counter()
                    s = self._conn()
                    s.sendall(json.dumps({"op": "put", "key": key, "len": len(data)}).encode() + b"\n")
                    s.sendall(data)
                    resp = json.loads(self.rf.readline())
                    if not resp.get("ok"):
                        raise OSError("put failed")
                    return time.perf_counter() - t0
            except OSError:
                self._reset()
                if attempt == retries - 1:
                    raise
                time.sleep(0.5)
        raise OSError("unreachable")

    def get(self, key: str, retries: int = 3) -> bytes | None:
        for attempt in range(retries):
            try:
                with self.lock:
                    s = self._conn()
                    s.sendall(json.dumps({"op": "get", "key": key}).encode() + b"\n")
                    resp = json.loads(self.rf.readline())
                    if not resp.get("ok"):
                        return None
                    return self.rf.read(resp["len"])
            except OSError:
                self._reset()
                if attempt == retries - 1:
                    raise
                time.sleep(0.5)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7400)
    a = ap.parse_args()
    serve(a.root, a.host, a.port)
