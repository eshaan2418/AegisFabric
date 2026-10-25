"""Post-training methods compared at equal analytic compute: continued SFT, iterative DPO, GRPO, reward model.

Own implementations (TRL not used: our tool-interleaved environment and char model don't fit its APIs).
All methods start from the same warm-start weights; each stops when its Compute ledger reaches `budget`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from data.synth import qa
from posttrain.model import DEVICE, ModelConfig, TinyLM, pad_batch
from posttrain.task import Compute, rollout, sft_batches, train_steps


def verifiable_reward(r: dict) -> float:
    return float(r["exact"]) + 0.1 * float(r["tool_ok"])


def verifiable_batch(outs: list[dict], c: Compute) -> np.ndarray:
    return np.array([verifiable_reward(o) for o in outs], float)


def continued_sft(model: TinyLM, seed: int, budget: float, lr: float = 1e-3, batch: int = 128) -> Compute:
    rng = np.random.default_rng(seed)
    eps = qa.make_split(4000, seed=20_000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    c = Compute()
    train_steps(model, opt, sft_batches(eps, batch, rng), c, budget_flops=budget)
    return c


def _masked_lp(model, seqs, masks):
    inp, tgt, m = pad_batch(seqs, masks)
    lp = torch.log_softmax(model(inp).float(), -1).gather(-1, tgt[..., None])[..., 0]
    return lp, m, int((tgt != 0).sum())


def dpo(
    model: TinyLM,
    seed: int,
    budget: float,
    beta: float = 0.1,
    lr: float = 3e-4,
    prompts: int = 64,
    k: int = 4,
    temperature: float = 1.0,
    reward_fn=verifiable_reward,
) -> tuple[Compute, dict]:
    """Iterative (online) DPO: sample k completions/prompt from current policy, pair best vs worst by reward,
    take one DPO gradient step per round against the fixed warm-start reference."""
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    c = Compute()
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    pool = qa.make_split(5_000, seed=30_000 + seed)
    rounds, n_pairs, t0 = 0, 0, time.perf_counter()
    while c.flops < budget:
        sel = [pool[i] for i in rng.choice(len(pool), prompts, replace=False)]
        eps = [e for e in sel for _ in range(k)]
        outs = rollout(model, eps, c, temperature, g)
        ch, rj = [], []
        for p in range(prompts):
            grp = outs[p * k : (p + 1) * k]
            rw = [reward_fn(o) for o in grp]
            if max(rw) > min(rw):
                ch.append(grp[int(np.argmax(rw))])
                rj.append(grp[int(np.argmin(rw))])
        rounds += 1
        if not ch:
            continue
        n_pairs += len(ch)
        model.train()
        seqs = [o["ids"] for o in ch + rj]
        masks = [o["gen_mask"] for o in ch + rj]
        with torch.no_grad():
            rlp, m, ntok = _masked_lp(ref, seqs, masks)
            ref_s = (rlp * m).sum(-1)
        c.add_fwd(ref, ntok)
        lp, m, ntok = _masked_lp(model, seqs, masks)
        pol_s = (lp * m).sum(-1)
        n = len(ch)
        logits = beta * ((pol_s[:n] - ref_s[:n]) - (pol_s[n:] - ref_s[n:]))
        loss = -Fn.logsigmoid(logits).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        c.add_train(model, ntok)
    c.wall_s += time.perf_counter() - t0
    return c, {"rounds": rounds, "pairs": n_pairs}


def grpo(
    model: TinyLM,
    seed: int,
    budget: float,
    lr: float = 3e-4,
    prompts: int = 32,
    group: int = 8,
    kl_beta: float = 0.02,
    temperature: float = 1.0,
    reward_batch=verifiable_batch,
) -> tuple[Compute, dict]:
    """GRPO (Shao et al. 2024): group-normalised advantages, one on-policy step per batch (ratio = 1, so
    the clipped surrogate reduces to the policy gradient), k3 KL estimator to the warm-start reference."""
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    c = Compute()
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    pool = qa.make_split(5_000, seed=40_000 + seed)
    iters, mean_r, t0 = 0, [], time.perf_counter()
    while c.flops < budget:
        sel = [pool[i] for i in rng.choice(len(pool), prompts, replace=False)]
        eps = [e for e in sel for _ in range(group)]
        outs = rollout(model, eps, c, temperature, g)
        r = reward_batch(outs, c).reshape(prompts, group)
        mean_r.append(float(r.mean()))
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)
        iters += 1
        keep = [i for i in range(len(outs)) if abs(adv.reshape(-1)[i]) > 1e-6]
        if not keep:
            continue
        seqs = [outs[i]["ids"] for i in keep]
        masks = [outs[i]["gen_mask"] for i in keep]
        a = torch.tensor(adv.reshape(-1)[keep], dtype=torch.float, device=DEVICE)
        model.train()
        with torch.no_grad():
            rlp, m, ntok = _masked_lp(ref, seqs, masks)
        c.add_fwd(ref, ntok)
        lp, m, ntok = _masked_lp(model, seqs, masks)
        pg = -(a[:, None] * lp * m).sum() / m.sum()
        d = rlp - lp
        kl = ((torch.exp(d) - d - 1) * m).sum() / m.sum()
        loss = pg + kl_beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        c.add_train(model, ntok)
    c.wall_s += time.perf_counter() - t0
    return c, {"iters": iters, "mean_train_reward_first": mean_r[:3], "mean_train_reward_last": mean_r[-3:]}


class RewardModel:
    """Scalar-head TinyLM initialised from the policy trunk; predicts P(exact-correct) from the trajectory."""

    def __init__(self, base: TinyLM):
        cfg = ModelConfig(**{**base.cfg.__dict__, "value_head": True})
        self.net = TinyLM(cfg).to(DEVICE)
        sd = {k: v for k, v in base.state_dict().items() if not k.startswith("head.")}
        self.net.load_state_dict(sd, strict=False)

    def scores(self, outs: list[dict]) -> np.ndarray:
        self.net.eval()
        res = []
        with torch.no_grad():
            for s in range(0, len(outs), 256):
                chunk = outs[s : s + 256]
                inp, tgt, _ = pad_batch([o["ids"] for o in chunk])
                v = self.net(inp)
                last = torch.tensor([len(o["ids"]) - 1 for o in chunk], device=v.device)
                res.append(
                    torch.sigmoid(v[torch.arange(len(chunk), device=v.device), last]).float().cpu().numpy()
                )
        return np.concatenate(res)

    def fit(
        self, outs: list[dict], c: Compute, epochs: int = 3, lr: float = 5e-4, batch: int = 64, seed: int = 0
    ):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        y_all = np.array([float(o["exact"]) for o in outs])
        self.net.train()
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(len(outs)), max(1, len(outs) // batch)):
                chunk = [outs[i] for i in idx]
                inp, tgt, _ = pad_batch([o["ids"] for o in chunk])
                v = self.net(inp)
                last = torch.tensor([len(o["ids"]) - 1 for o in chunk], device=v.device)
                logit = v[torch.arange(len(chunk), device=v.device), last]
                y = torch.tensor(y_all[idx], dtype=torch.float, device=v.device)
                loss = Fn.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                c.add_train(self.net, int((tgt != 0).sum()))


def train_reward_model(
    policy: TinyLM, seed: int, n_train: int = 3000, n_test: int = 1000
) -> tuple[RewardModel, dict]:
    c = Compute()
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    eps = qa.make_split(n_train + n_test, seed=50_000 + seed)
    outs = rollout(policy, eps, c, 1.0, g)
    rm = RewardModel(policy)
    rm.fit(outs[:n_train], c, seed=seed)
    s = rm.scores(outs[n_train:])
    y = np.array([o["exact"] for o in outs[n_train:]], float)
    from sklearn.metrics import roc_auc_score

    acc = float(((s > 0.5) == (y > 0.5)).mean())
    auc = float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")
    c.wall_s += time.perf_counter() - t0
    return rm, {
        "rm_test_accuracy": acc,
        "rm_test_auc": auc,
        "rm_test_pos_rate": float(y.mean()),
        "rm_compute": c.as_dict(),
    }


def grpo_rm(model: TinyLM, rm: RewardModel, seed: int, budget: float, **kw) -> tuple[Compute, dict]:
    """GRPO with the learned RM's P(correct) as reward (+0.1 verifiable tool validity).
    RM forward passes over every rollout are charged to the compute ledger."""

    def reward_batch(outs: list[dict], c: Compute) -> np.ndarray:
        c.add_fwd(rm.net, sum(len(o["ids"]) for o in outs))
        return rm.scores(outs) + 0.1 * np.array([float(o["tool_ok"]) for o in outs])

    return grpo(model, seed, budget, reward_batch=reward_batch, **kw)
