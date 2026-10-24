"""Launch / kill `aegis-ckpt serve` processes (tests, benchmarks, crash campaign, lab orchestration)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def find_binary() -> str:
    env = os.environ.get("AEGIS_CKPT_BIN")
    if env:
        return env
    for p in (_HERE / "target" / "release" / "aegis-ckpt", _HERE / "target" / "debug" / "aegis-ckpt"):
        if p.exists():
            return str(p)
    w = shutil.which("aegis-ckpt")
    if w:
        return w
    raise FileNotFoundError("aegis-ckpt binary not found; run `cargo build --release` in ckpt/")


class ServerProcess:
    """`aegis-ckpt serve` as a child process. `start()` blocks until recovery is done and it listens."""

    def __init__(
        self,
        root: str | Path,
        listen: str = "127.0.0.1:0",
        io: str = "pwrite",
        io_threads: int = 4,
        netns: str | None = None,
        binary: str | None = None,
    ):
        self.root = Path(root)
        self.listen = listen
        self.io = io
        self.io_threads = io_threads
        self.netns = netns
        self.binary = binary or find_binary()
        self.proc: subprocess.Popen | None = None
        self.addr: str | None = None
        self.recovery: dict = {}
        self.startup_s = 0.0

    def start(self, timeout: float = 120.0) -> ServerProcess:
        cmd = [
            self.binary,
            "serve",
            "--root",
            str(self.root),
            "--listen",
            self.listen,
            "--io",
            self.io,
            "--io-threads",
            str(self.io_threads),
        ]
        if self.netns:
            cmd = ["ip", "netns", "exec", self.netns, *cmd]
        t0 = time.perf_counter()
        # stderr goes to a temp file (an unread pipe could fill up and block the server)
        self._err = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, text=True)
        deadline = time.time() + timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                self.proc.wait()
                self._err.seek(0)
                raise RuntimeError(f"aegis-ckpt exited during startup: {self._err.read()[-2000:]}")
            if line.startswith("RECOVERY "):
                self.recovery = json.loads(line[len("RECOVERY ") :])
            elif line.startswith("READY "):
                self.addr = line.split()[1]
                self.startup_s = time.perf_counter() - t0
                return self
        raise TimeoutError("aegis-ckpt did not become ready")

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def kill9(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)
            self.proc.wait()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.kill9()

    def __enter__(self) -> ServerProcess:
        return self.start()

    def __exit__(self, *a) -> None:
        self.stop()
