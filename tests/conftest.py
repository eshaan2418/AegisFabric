import shutil

import pytest


class Engine:
    """A running aegis-ckpt server on a store directory (kill -9 / restart for crash tests)."""

    def __init__(self, root):
        self.root = root
        self.proc = None
        self.addr = None
        self.info = None

    def start(self) -> str:
        from ckpt import client as cc

        self.proc, self.addr, self.info = cc.start_server(self.root, io="auto", seg_mb=4)
        return self.addr

    def kill(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def restart(self) -> str:
        self.kill()
        return self.start()


@pytest.fixture
def engine(tmp_path):
    from ckpt import client as cc

    if shutil.which("cargo") is None and not cc.binary().exists():
        pytest.skip("aegis-ckpt binary not built and cargo unavailable")
    e = Engine(tmp_path / "store")
    e.start()
    yield e
    e.kill()
