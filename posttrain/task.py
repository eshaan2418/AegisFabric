"""Task plumbing: tokenised episodes, compute accounting, evaluation, pretraining and SFT loops."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from data.synth import finance as F
from data.synth import qa
from posttrain.model import STOI, TinyLM, encode, flops_per_token, generate, pad_batch


@dataclasses.dataclass
class Compute:
    """Analytic compute ledger (FLOPs ≈ 2N/token forward, 6N/token forward+backward) + wall-clock."""

    flops: float = 0.0
    train_tokens: int = 0
    fwd_tokens: int = 0
    wall_s: float = 0.0

    def add_train(self, model, n_tokens: int) -> None:
        self.flops += flops_per_token(model, True) * n_tokens
        self.train_tokens += n_tokens

    def add_fwd(self, model, n_tokens: int) -> None:
        self.flops += flops_per_token(model, False) * n_tokens
        self.fwd_tokens += n_tokens

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def episode_tokens(ep: qa.Episode) -> tuple[list[int], list[int]]:
    """Token ids for prompt+gold completion and a loss mask: 1 on model-written tokens, 0 on prompt and R: lines."""
    ids = encode(ep.prompt)
    mask = [0] * len(ids)
    for line in ep.gold_completion.split("\n")[:-1]:
        t = encode(line + "\n")
        ids += t
        mask += [0 if line.startswith("R: ") else 1] * len(t)
    return ids, mask


def lm_windows(rng: np.random.Generator, n: int, max_len: int) -> list[list[int]]:
    """Public pretraining corpus: random windows of synthetic statement text and call transcripts."""
    out = []
    while len(out) < n:
        st = F.make_statement(rng)
        text = st.to_text() + "\n"
        if rng.random() < 0.5:
            text += "\n".join(t["text"] for t in F.make_transcript(rng, st)) + "\n"
        ids = encode(text)
        if len(ids) > max_len:
            s = int(rng.integers(0, len(ids) - max_len))
            ids = ids[s : s + max_len]
        out.append(ids)
    return out


def train_steps(
    model: TinyLM,
    opt,
    batches,
    compute: Compute,
    budget_flops: float | None = None,
    max_steps: int | None = None,
    clip: float = 1.0,
) -> list[float]:
    """Masked next-token CE over an iterator of (ids, masks) lists. Stops at budget or max_steps."""
    model.train()
    losses = []
    t0 = time.perf_counter()
    for step, (seqs, masks) in enumerate(batches):
        if max_steps is not None and step >= max_steps:
            break
        if budget_flops is not None and compute.flops >= budget_flops:
            break
        inp, tgt, m = pad_batch(seqs, masks)
        logits = model(inp)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
        )
        loss = (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        compute.add_train(model, int((tgt != 0).sum()))
        losses.append(loss.item())
    compute.wall_s += time.perf_counter() - t0
    return losses


def sft_batches(episodes: list[qa.Episode], batch: int, rng: np.random.Generator):
    toks = [episode_tokens(e) for e in episodes]
    while True:
        idx = rng.permutation(len(toks))
        for s in range(0, len(idx) - batch + 1, batch):
            sel = [toks[i] for i in idx[s : s + batch]]
            yield [a for a, _ in sel], [b for _, b in sel]


def lm_batches(rng: np.random.Generator, batch: int, max_len: int):
    while True:
        w = lm_windows(rng, batch, max_len)
        yield w, [[1] * len(x) for x in w]


def make_env(ep: qa.Episode):
    led = qa.Ledger(ep.statement)
    return led.run


def rollout(
    model: TinyLM,
    episodes: list[qa.Episode],
    compute: Compute | None,
    temperature: float,
    generator: torch.Generator | None = None,
    batch: int = 128,
) -> list[dict]:
    outs = []
    for s in range(0, len(episodes), batch):
        eps = episodes[s : s + batch]
        res = generate(
            model,
            [e.prompt for e in eps],
            [make_env(e) for e in eps],
            temperature=temperature,
            generator=generator,
        )
        for e, r in zip(eps, res, strict=True):
            r.update(qa.score(e, r["text"]))
            outs.append(r)
            if compute is not None:
                compute.add_fwd(model, len(r["ids"]))
    return outs


def evaluate(model: TinyLM, episodes: list[qa.Episode]) -> dict:
    res = rollout(model, episodes, None, temperature=0.0)
    by_type: dict[str, list[bool]] = {}
    for e, r in zip(episodes, res, strict=True):
        by_type.setdefault(e.qtype, []).append(r["exact"])
    return {
        "exact": float(np.mean([r["exact"] for r in res])),
        "tool_ok": float(np.mean([r["tool_ok"] for r in res])),
        "by_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "n": len(res),
    }


def warm_start(
    seed: int, cfg, n_sft: int, pretrain_steps: int, sft_steps: int, batch: int = 32, lr: float = 2e-3
) -> tuple[TinyLM, Compute, dict]:
    """Stage 0 (LM pretraining on public synthetic corpus) + stage 1 (SFT on n_sft labelled episodes)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyLM(cfg)
    compute = Compute()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pre = train_steps(model, opt, lm_batches(rng, batch, cfg.max_len), compute, max_steps=pretrain_steps)
    sft_eps = qa.make_split(n_sft, seed=10_000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sft = train_steps(model, opt, sft_batches(sft_eps, batch, rng), compute, max_steps=sft_steps)
    info = {
        "pretrain_final_loss": float(np.mean(pre[-20:])),
        "sft_final_loss": float(np.mean(sft[-20:])),
        "n_params": model.n_params(),
    }
    return model, compute, info


assert STOI["\n"] != 0
