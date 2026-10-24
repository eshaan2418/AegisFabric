"""Tiny multimodal (chart image + byte-level text) decoder-only transformer.

Sequence layout: [P image-patch tokens] + [T text tokens]; causal attention over the whole sequence
(image tokens come first, so every text token can see the full image). Loss = next-byte prediction on text.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelConfig:
    vocab: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    img_size: int = 32
    patch: int = 8
    text_len: int = 128
    dropout: float = 0.0

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch) ** 2

    @property
    def seq_len(self) -> int:
        return self.n_patches + self.text_len


PRESETS = {
    "tiny": ModelConfig(d_model=128, n_layers=4, n_heads=4, d_ff=512),
    "small": ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024),
    "base": ModelConfig(d_model=384, n_layers=8, n_heads=8, d_ff=1536),
}


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_ff)
        self.fc2 = nn.Linear(c.d_ff, c.d_model)
        self.n_heads = c.n_heads
        self.drop = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0
        )
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyVLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.patch_embed = nn.Linear(c.patch * c.patch, c.d_model)
        self.tok_embed = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Parameter(torch.zeros(1, c.seq_len, c.d_model))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        nn.init.normal_(self.pos, std=0.02)

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, H, W) float in [0,1] -> (B, P, patch*patch)
        b, h, w = img.shape
        p = self.cfg.patch
        x = img.view(b, h // p, p, w // p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(b, (h // p) * (w // p), p * p)

    def forward(self, img: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Returns logits for text positions: (B, T, vocab). tokens: (B, T) int64."""
        pe = self.patch_embed(self.patchify(img))
        te = self.tok_embed(tokens)
        x = torch.cat([pe, te], dim=1) + self.pos[:, : pe.shape[1] + te.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x[:, pe.shape[1] :])
        return self.head(x)

    def loss(self, img: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(img, tokens[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=0
        )


def count_params(m: nn.Module, non_embedding: bool = False) -> int:
    n = sum(p.numel() for p in m.parameters())
    if non_embedding and isinstance(m, TinyVLM):
        n -= m.tok_embed.weight.numel() + m.pos.numel()
    return n


def flops_per_token(c: ModelConfig, n_params_nonembed: int) -> float:
    """Training FLOPs per token (forward+backward), PaLM-style: 6N + 12·L·d·S (attention scores)."""
    return 6.0 * n_params_nonembed + 12.0 * c.n_layers * c.d_model * c.seq_len
