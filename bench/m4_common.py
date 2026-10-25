"""Shared helpers for the M4/M5 checkpoint experiments (bench/m4_*.py, bench/m5_envelope.py).

`Trainer`: the real TinyVLM (train/model.py) + AdamW on the synthetic chart/text stream (train/data.py),
so the deltas between successive checkpoints are those of actual training steps.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch


class Trainer:
    def __init__(self, preset: str = "base", seed: int = 0, batch: int = 8, freeze: str = "none", lr=1e-3):
        from train.data import ShardedStream
        from train.model import PRESETS, TinyVLM

        torch.manual_seed(seed)
        self.cfg = PRESETS[preset]
        self.model = TinyVLM(self.cfg)
        if freeze == "half":
            # fine-tuning-style: frozen patch/token/position embeddings and the lower half of the blocks
            frozen = ["patch_embed", "tok_embed", "pos"] + [
                f"blocks.{i}." for i in range(self.cfg.n_layers // 2)
            ]
            for n, p in self.model.named_parameters():
                if any(n == f or n.startswith(f) for f in frozen):
                    p.requires_grad_(False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        self.stream = ShardedStream(1234 + seed, 0, 1, batch, self.cfg.text_len, self.cfg.img_size)
        self.step_no = 0

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
        """Full model + optimizer state exactly as the trainer checkpoints it (cpu_offload fix on main)."""
        from train.trainer import get_full_state

        return get_full_state(self.model, self.opt)

    def tensors(self) -> dict[str, torch.Tensor]:
        """Tensor leaves of state(), flat names."""
        from train.checkpointing import flatten

        return {
            k.replace("\x1f", "/"): v for k, v in flatten(self.state()).items() if isinstance(v, torch.Tensor)
        }


def loadavg() -> list[float]:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return list(os.getloadavg())
