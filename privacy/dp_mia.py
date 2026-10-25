"""DP-SGD (Opacus) on the private tool-use fine-tuning step + membership-inference attacks on canaries.

Definitions
- Base model: char LM pretrained only on the *public* synthetic corpus (statements + transcripts).
- Private fine-tuning step (the step that is made DP): supervised fine-tuning on D_priv =
  `n_task` private tool-use episodes ∪ M member canaries. Each record (episode or canary) is one example;
  DP is example-level.
- Canaries: `note SYN-dddddddd pin dddd-dddd` with random digits. Canary pool of 2M is split 50/50 into
  members (each inserted once into D_priv) and non-members (never trained on). Split is seeded.
- Attacks: (a) loss threshold (Yeom et al. 2018), score = −loss; (b) offline LiRA (Carlini et al. 2022):
  K shadow models trained with the same recipe on fresh task data and *no* canaries; per-canary OUT
  Gaussian fit of the logit-scaled confidence; score = −log-likelihood under OUT.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score, roc_curve

from data.synth import qa
from posttrain.model import DEVICE, ModelConfig, TinyLM, encode
from posttrain.task import Compute, episode_tokens, lm_batches, train_steps


def make_canaries(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        acct = "".join(str(d) for d in rng.integers(0, 10, 8))
        pin = (
            "".join(str(d) for d in rng.integers(0, 10, 4))
            + "-"
            + "".join(str(d) for d in rng.integers(0, 10, 4))
        )
        out.append(f"note SYN-{acct} pin {pin}\n")
    return out


def pretrain_base(seed: int, cfg: ModelConfig, steps: int = 300, batch: int = 32) -> TinyLM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = TinyLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    train_steps(m, opt, lm_batches(rng, batch, cfg.max_len), Compute(), max_steps=steps)
    return m


def build_dataset(task_eps: list[qa.Episode], canaries: list[str], max_len: int):
    """Fixed-length padded tensors (inp, tgt, mask) for one example per record."""
    recs = [episode_tokens(e) for e in task_eps] + [(encode(c), [1] * len(c)) for c in canaries]
    T = min(max_len, max(len(r[0]) for r in recs))
    n = len(recs)
    inp = torch.zeros((n, T), dtype=torch.long)
    tgt = torch.zeros((n, T), dtype=torch.long)
    msk = torch.zeros((n, T), dtype=torch.float)
    for i, (ids, m) in enumerate(recs):
        ids, m = ids[:T], m[:T]
        full = [encode("\n")[0]] + ids
        inp[i, : len(ids)] = torch.tensor(full[:-1])
        tgt[i, : len(ids)] = torch.tensor(ids)
        msk[i, : len(ids)] = torch.tensor(m, dtype=torch.float)
    return torch.utils.data.TensorDataset(inp, tgt, msk)


def per_example_loss(model, inp, tgt, msk) -> torch.Tensor:
    logits = model(inp)
    ce = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(
        tgt.shape
    )
    return (ce * msk).sum(1) / msk.sum(1).clamp(min=1)


def finetune(
    base: TinyLM,
    ds,
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
    epsilon: float | None,
    delta: float,
    max_grad_norm: float,
    accountant: str = "rdp",
    clip_only: bool = False,
) -> tuple[TinyLM, dict]:
    """Fine-tune a copy of `base` on `ds`. epsilon=None → non-private (plain AdamW, same batch/epochs).

    clip_only=True (with epsilon=None) is the ablation "DP-SGD mechanics without noise": Opacus Poisson
    sampling + per-sample clipping with noise_multiplier=0 (no privacy guarantee, ε = ∞)."""
    import copy

    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    model.train()
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    info: dict = {
        "epsilon_target": epsilon,
        "delta": delta,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "n_records": len(ds),
        "sample_rate": batch / len(ds),
    }
    engine = None
    if epsilon is not None:
        from opacus import PrivacyEngine

        engine = PrivacyEngine(accountant=accountant)
        model, opt, dl = engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=dl,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
        )
        info.update(
            noise_multiplier=float(opt.noise_multiplier), max_grad_norm=max_grad_norm, accountant=accountant
        )
    elif clip_only:
        from opacus import PrivacyEngine

        engine = PrivacyEngine(accountant=accountant)
        model, opt, dl = engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=dl,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
        )
        info.update(noise_multiplier=0.0, max_grad_norm=max_grad_norm, clip_only=True)
    steps = 0
    for _ in range(epochs):
        for inp, tgt, msk in dl:
            if len(inp) == 0:
                continue
            inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
            loss = per_example_loss(model, inp, tgt, msk).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if engine is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
    info["steps"] = steps
    if engine is not None and clip_only:
        model = model._module
    elif engine is not None:
        info["epsilon_spent_rdp"] = float(engine.get_epsilon(delta))
        try:  # independent cross-check with the PRV accountant on the same (q, sigma, steps)
            from opacus.accountants import PRVAccountant

            prv = PRVAccountant()
            prv.history = [(float(opt.noise_multiplier), info["sample_rate"], steps)]
            info["epsilon_spent_prv"] = float(prv.get_epsilon(delta))
        except Exception as e:  # pragma: no cover
            info["epsilon_spent_prv"] = f"unavailable: {e}"
        model = model._module  # unwrap GradSampleModule
    model.eval()
    return model, info


@torch.no_grad()
def canary_losses(model: TinyLM, canaries: list[str]) -> np.ndarray:
    ds = build_dataset([], canaries, model.cfg.max_len)
    inp, tgt, msk = (t.to(DEVICE) for t in ds.tensors)
    return per_example_loss(model, inp, tgt, msk).float().cpu().numpy()


def attack_metrics(scores: np.ndarray, is_member: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    auc = float(roc_auc_score(is_member, scores))
    fpr, tpr, _ = roc_curve(is_member, scores)
    tpr_at = {f"tpr@{int(f * 100)}%fpr": float(np.interp(f, fpr, tpr)) for f in (0.01, 0.05, 0.1)}
    rng = np.random.default_rng(seed)
    boots = []
    n = len(scores)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if 0 < is_member[i].mean() < 1:
            boots.append(roc_auc_score(is_member[i], scores[i]))
    return {
        "auc": auc,
        "auc_boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        **tpr_at,
    }


def lira_offline_scores(
    target_loss: np.ndarray, shadow_losses: np.ndarray, global_var: bool = False
) -> np.ndarray:
    """Offline LiRA. Confidence φ = log(p/(1−p)) with p = exp(−loss) (mean per-token prob proxy).
    shadow_losses: (K, n) OUT losses. Score = one-sided Gaussian test that φ_target exceeds OUT."""
    from scipy.stats import norm

    def phi(loss):
        p = np.clip(np.exp(-loss), 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p)

    out = phi(shadow_losses)
    mu = out.mean(0)
    if global_var:  # fixed variance pooled over canaries: recommended when few shadows (Carlini et al. 2022)
        sd = np.full_like(mu, (out - mu).std(ddof=1) + 1e-3)
    else:
        sd = out.std(0, ddof=1) + 1e-3
    return norm.logcdf(phi(target_loss), loc=mu, scale=sd)  # higher = more member-like
