"""Synthetic multimodal training data: (running-balance chart image, statement text bytes).

Examples are generated deterministically from (seed, global index), so every rank can produce its shard
without any shared files, and a restarted job can resume at an exact sample offset."""

from __future__ import annotations

import numpy as np
import torch

from data.synth.finance import make_statement, render_chart


def example(seed: int, idx: int, text_len: int, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, idx])
    st = make_statement(rng)
    img = render_chart(st, img_size).astype(np.float32) / 255.0
    raw = st.to_text().encode("ascii", "replace")[: text_len - 1]
    toks = np.zeros(text_len, dtype=np.int64)  # 0 = pad (ignored in loss)
    toks[0] = 2  # BOS
    toks[1 : 1 + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return img, toks


class ShardedStream:
    """Infinite deterministic stream of batches for one data-parallel rank."""

    def __init__(
        self,
        seed: int,
        dp_rank: int,
        dp_world: int,
        batch: int,
        text_len: int,
        img_size: int,
        start_step: int = 0,
        cache: int = 4096,
    ):
        self.seed, self.dp_rank, self.dp_world = seed, dp_rank, dp_world
        self.batch, self.text_len, self.img_size = batch, text_len, img_size
        self.step = start_step
        # Precompute a pool of examples once (generation is Python-heavy); batches index into it.
        self.pool_n = cache
        imgs, toks = zip(*(example(seed, i, text_len, img_size) for i in range(cache)), strict=True)
        self.imgs = torch.from_numpy(np.stack(imgs))
        self.toks = torch.from_numpy(np.stack(toks))

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.step * self.dp_world * self.batch + self.dp_rank * self.batch
        idx = (torch.arange(self.batch) + g) % self.pool_n
        self.step += 1
        return self.imgs[idx], self.toks[idx]


def eval_batch(seed: int, n: int, text_len: int, img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Held-out examples (index space disjoint from the training pool)."""
    imgs, toks = zip(*(example(seed + 10_000, i, text_len, img_size) for i in range(n)), strict=True)
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(toks))
