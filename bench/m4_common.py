"""Shared helpers for the M4 checkpoint-engine experiments (bench/m4_*.py).

* `Trainer`: the real TinyVLM (`train/model.py`) + AdamW on the synthetic chart/text stream, so checkpoint
  deltas between successive saves are those of actual training steps.
* `LoopFS`: an ext4 or XFS filesystem on a loopback image inside the privileged lab container. The image
  file lives on the container's overlay filesystem, i.e. on the Docker Desktop VM's virtual disk (vda),
  so absolute numbers include the VM's virtio block layer and the macOS host's APFS underneath.
* Device-level write accounting (`/sys/block/loopN/stat`, sectors written) for write amplification.
"""

from __future__ import annotations

import io
import os
import subprocess
import time
from pathlib import Path

import torch

LAB_BASE = Path("/var/tmp/aegis-m4")


def in_lab() -> bool:
    return os.environ.get("AEGIS_HOST_LABEL") == "lab-container" or Path("/.dockerenv").exists()


class Trainer:
    def __init__(self, preset: str = "small", seed: int = 0, batch: int = 8, freeze: str = "none", lr=1e-3):
        from train.data import ShardedStream
        from train.model import PRESETS, TinyVLM

        torch.manual_seed(seed)
        self.cfg = PRESETS[preset]
        self.model = TinyVLM(self.cfg)
        if freeze == "half":
            # fine-tuning-style: frozen patch embedding, token embedding and the lower half of the blocks
            frozen = ["patch_embed", "tok_embed", "pos"] + [f"blocks.{i}." for i in range(self.cfg.n_layers // 2)]
            for n, p in self.model.named_parameters():
                if any(n == f or n.startswith(f) for f in frozen):
                    p.requires_grad_(False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.stream = ShardedStream(1234 + seed, 0, 1, batch, self.cfg.text_len, self.cfg.img_size)
        self.step_no = 0

    def steps(self, n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            self.one_step()
        return time.perf_counter() - t0

    def one_step(self) -> float:
        t0 = time.perf_counter()
        imgs, toks = self.stream.next()
        loss = self.model.loss(imgs, toks)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.step_no += 1
        return time.perf_counter() - t0

    def state(self) -> dict:
        from train.trainer import get_full_state

        return get_full_state(self.model, self.opt)


def serialize(state) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def sh(cmd: str, check: bool = True) -> str:
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True).stdout.strip()


class LoopFS:
    """mkfs + mount a loopback image. Requires root (privileged container)."""

    def __init__(self, kind: str, size_gb: int = 6):
        assert kind in ("ext4", "xfs")
        self.kind, self.size_gb = kind, size_gb
        self.img = LAB_BASE / f"{kind}.img"
        self.mnt = Path(f"/mnt/m4-{kind}")
        self.dev = ""

    def __enter__(self) -> LoopFS:
        LAB_BASE.mkdir(parents=True, exist_ok=True)
        sh(f"umount {self.mnt} 2>/dev/null; true", check=False)
        self.img.unlink(missing_ok=True)
        sh(f"truncate -s {self.size_gb}G {self.img}")
        mk = "mkfs.ext4 -q -F" if self.kind == "ext4" else "mkfs.xfs -q -f"
        sh(f"{mk} {self.img}")
        self.mnt.mkdir(parents=True, exist_ok=True)
        self.dev = sh(f"losetup --find --show {self.img}")
        sh(f"mount {self.dev} {self.mnt}")
        return self

    def __exit__(self, *a) -> None:
        sh(f"umount {self.mnt}", check=False)
        sh(f"losetup -d {self.dev}", check=False)
        self.img.unlink(missing_ok=True)

    def written_bytes(self) -> int:
        """Bytes written to the loop block device so far (includes journal + metadata)."""
        name = Path(self.dev).name
        f = Path(f"/sys/block/{name}/stat").read_text().split()
        return int(f[6]) * 512

    def settle(self) -> None:
        os.sync()


def drop_caches() -> None:
    """Cold-cache reads (lab container is privileged; drops the VM's page cache)."""
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
    except OSError:
        pass


def proc_write_bytes(pid: int | None = None) -> int:
    p = Path(f"/proc/{pid or 'self'}/io")
    for line in p.read_text().splitlines():
        if line.startswith("write_bytes:"):
            return int(line.split()[1])
    return 0


def loadavg() -> list[float]:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return list(os.getloadavg())


def fsync_path(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
