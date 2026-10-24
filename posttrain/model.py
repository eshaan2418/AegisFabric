"""Small char-level causal transformer (own implementation) with KV-cached, tool-interleaved generation.

Built only from nn.Embedding / nn.Linear / nn.LayerNorm so Opacus can compute per-sample gradients.
"""

from __future__ import annotations

import dataclasses
import math
import os
import string

import torch
import torch.nn as nn
import torch.nn.functional as Fn

CHARS = "\x00" + string.ascii_letters + string.digits + " .,:;?!()+-_=/'\n[]<>#%&*"
STOI = {c: i for i, c in enumerate(CHARS)}
PAD = 0
VOCAB = len(CHARS)
# Compute device for all M8 code paths ("cpu" or "mps"); sampling always happens on CPU for determinism.
DEVICE = torch.device(os.environ.get("AEGIS_DEVICE", "cpu"))


def encode(s: str) -> list[int]:
    return [STOI.get(c, STOI["#"]) for c in s]


def decode(ids) -> str:
    return "".join(CHARS[i] for i in ids if i != PAD)


@dataclasses.dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 256
    vocab: int = VOCAB
    value_head: bool = False  # reward model: scalar head on last token


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x, mask=None, cache=None):
        B, T, D = x.shape
        hd = D // self.h
        y = self.ln1(x)
        q = self.q(y).view(B, T, self.h, hd).transpose(1, 2)
        k = self.k(y).view(B, T, self.h, hd).transpose(1, 2)
        v = self.v(y).view(B, T, self.h, hd).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(z)
        x = x + self.fc2(Fn.gelu(self.fc1(self.ln2(x))))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1 if cfg.value_head else cfg.vocab)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) -> logits (B, T, V) or values (B, T)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)[None].expand(B, T)  # batch-first for Opacus per-sample grads
        x = self.tok(idx) + self.pos(pos)
        mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()[None, None]
        for b in self.blocks:
            x = b(x, mask)
        out = self.head(self.ln_f(x))
        return out.squeeze(-1) if self.cfg.value_head else out

    @torch.no_grad()
    def step(self, tok: torch.Tensor, pos: int, caches: list[dict]) -> torch.Tensor:
        """One incremental decoding step. tok: (B,) -> logits (B, V)."""
        x = self.tok(tok[:, None]) + self.pos.weight[pos][None, None]
        for b, c in zip(self.blocks, caches, strict=True):
            x = b(x, None, c)
        return self.head(self.ln_f(x))[:, 0]


def flops_per_token(model: nn.Module, train: bool) -> float:
    """Standard approximation: 2N per token forward, 6N forward+backward (attention term ignored; T<=256)."""
    n = sum(p.numel() for p in model.parameters())
    return (6.0 if train else 2.0) * n


@torch.no_grad()
def generate(
    model: TinyLM,
    prompts: list[str],
    env_fns: list,
    temperature: float = 0.0,
    max_new: int = 160,
    generator: torch.Generator | None = None,
) -> list[dict]:
    """Batched decoding with tool interleaving.

    All sequences start at position 0; prompt characters and tool results (`R: ...` lines) are *forced*
    tokens (not sampled, excluded from policy log-probs). `env_fns[i](call_line) -> result str | None`.
    Returns per sequence: ids (list[int]), gen_mask (1 where model-sampled), text (completion only),
    n_tokens (total positions processed).
    """
    model.eval()
    dev = next(model.parameters()).device
    B = len(prompts)
    caches = [dict() for _ in model.blocks]
    queues = [encode(p) for p in prompts]
    ids: list[list[int]] = [[] for _ in range(B)]
    gmask: list[list[int]] = [[] for _ in range(B)]
    plen = [len(q) for q in queues]
    line_buf = ["" for _ in range(B)]
    done = [False] * B
    max_len = model.cfg.max_len
    nl = STOI["\n"]
    prev = torch.full((B,), nl, dtype=torch.long, device=dev)  # BOS = newline
    total_steps = min(max_len, max(plen) + max_new)
    for pos in range(total_steps):
        logits = model.step(prev, pos, caches).float().cpu()
        if temperature > 0:
            probs = (logits / temperature).softmax(-1)
            sampled = torch.multinomial(probs, 1, generator=generator)[:, 0].tolist()
        else:
            sampled = logits.argmax(-1).tolist()
        nxt = sampled
        for i in range(B):
            if done[i]:
                nxt[i] = PAD
                continue
            forced = bool(queues[i])
            if forced:
                nxt[i] = queues[i].pop(0)
            t = nxt[i]
            ids[i].append(t)
            gmask[i].append(0 if forced else 1)
            if not forced and pos >= plen[i]:
                if t == nl:
                    line = line_buf[i]
                    line_buf[i] = ""
                    if line.startswith("A: "):
                        done[i] = True
                    elif line.startswith("C: "):
                        r = env_fns[i](line)
                        queues[i].extend(encode("R: " + (r if r is not None else "ERROR") + "\n"))
                else:
                    line_buf[i] += CHARS[t]
            elif forced and pos >= plen[i] and t == nl:
                line_buf[i] = ""
        if all(done):
            break
        prev = torch.tensor(nxt, dtype=torch.long, device=dev)
    out = []
    for i in range(B):
        out.append(
            {
                "ids": ids[i],
                "gen_mask": gmask[i],
                "text": decode(ids[i][plen[i] :]),
                "prompt_len": plen[i],
                "finished": done[i],
            }
        )
    return out


def pad_batch(seqs: list[list[int]], masks: list[list[int]] | None = None):
    """Right-pad to (B, T). Input = seq[:-1] (prefixed with BOS newline), target = seq."""
    T = max(len(s) for s in seqs)
    B = len(seqs)
    inp = torch.full((B, T), PAD, dtype=torch.long)
    tgt = torch.full((B, T), PAD, dtype=torch.long)
    m = torch.zeros((B, T), dtype=torch.float)
    for i, s in enumerate(seqs):
        full = [STOI["\n"]] + s
        inp[i, : len(s)] = torch.tensor(full[:-1])
        tgt[i, : len(s)] = torch.tensor(s)
        if masks is not None:
            m[i, : len(s)] = torch.tensor(masks[i], dtype=torch.float)
        else:
            m[i, : len(s)] = 1.0
    return inp.to(DEVICE), tgt.to(DEVICE), m.to(DEVICE)


def seq_logprobs(model: TinyLM, inp, tgt, m) -> torch.Tensor:
    """Sum of log-probs of masked target tokens per sequence: (B,)."""
    logits = model(inp)
    lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt[..., None])[..., 0]
    return (lp * m).sum(-1)


def token_logprobs(model: TinyLM, inp, tgt) -> torch.Tensor:
    logits = model(inp)
    return torch.log_softmax(logits.float(), -1).gather(-1, tgt[..., None])[..., 0]
