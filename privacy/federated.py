"""Federated fine-tuning simulation: 3 sites (TOR/MTL/US), non-IID tool-use data, FedAvg / FedProx.

In-process simulation: no network transport. Communication is *accounted* (bytes of fp32 parameters
exchanged per round), not measured on a wire. Site data never leaves its site object; only weights do.
Weights are not secret: federated learning alone gives no formal privacy guarantee (see DP-SGD).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from posttrain.model import TinyLM
from posttrain.task import Compute, sft_batches, train_steps

SITES = ["TOR", "MTL", "US"]


def _prox_steps(model, global_sd, opt, batches, steps, mu):
    """Local SGD steps with FedProx proximal term (mu/2)||w - w_global||^2 (mu=0 -> FedAvg)."""
    from posttrain.model import pad_batch

    model.train()
    gparams = {k: v.detach() for k, v in global_sd.items()}
    for _ in range(steps):
        seqs, masks = next(batches)
        inp, tgt, m = pad_batch(seqs, masks)
        logits = model(inp)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
        )
        loss = (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        if mu > 0:
            loss = loss + mu / 2 * sum(((p - gparams[n]) ** 2).sum() for n, p in model.named_parameters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def fed_train(
    base: TinyLM, site_data: dict, rounds: int, local_steps: int, batch: int, lr: float, mu: float, seed: int
) -> tuple[TinyLM, dict]:
    glob = copy.deepcopy(base)
    rng = np.random.default_rng(seed)
    iters = {s: sft_batches(site_data[s], batch, np.random.default_rng(rng.integers(1 << 31))) for s in SITES}
    sizes = np.array([len(site_data[s]) for s in SITES], float)
    w = sizes / sizes.sum()
    n_params = sum(p.numel() for p in glob.parameters())
    for _ in range(rounds):
        gsd = copy.deepcopy(glob.state_dict())
        locals_ = []
        for s in SITES:
            local = copy.deepcopy(glob)
            # fresh optimizer per round (stateless clients), as in standard FedAvg
            opt = torch.optim.AdamW(local.parameters(), lr=lr, weight_decay=0.0)
            _prox_steps(local, gsd, opt, iters[s], local_steps, mu)
            locals_.append(local.state_dict())
        avg = {k: sum(w[i] * locals_[i][k].float() for i in range(len(SITES))) for k in gsd}
        glob.load_state_dict(avg)
    comm = {
        "bytes_per_round": 2 * len(SITES) * n_params * 4,
        "rounds": rounds,
        "total_bytes": 2 * len(SITES) * n_params * 4 * rounds,
        "n_params": n_params,
    }
    return glob, comm


def central_train(base: TinyLM, eps, steps: int, batch: int, lr: float, seed: int) -> TinyLM:
    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    train_steps(m, opt, sft_batches(eps, batch, np.random.default_rng(seed)), Compute(), max_steps=steps)
    return m
