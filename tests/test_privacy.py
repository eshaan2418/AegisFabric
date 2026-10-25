import numpy as np
import torch

from data.synth import qa
from posttrain.model import ModelConfig, TinyLM
from privacy import dp_mia as D
from privacy.federated import fed_train

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=2, max_len=192)


def test_canaries_unique_and_seeded():
    a = D.make_canaries(50, 1)
    assert a == D.make_canaries(50, 1) and len(set(a)) == 50
    assert all(c.startswith("note SYN-") for c in a)


def test_attack_metrics_perfect_and_random():
    y = np.array([1] * 50 + [0] * 50)
    assert D.attack_metrics(y.astype(float), y)["auc"] == 1.0
    r = D.attack_metrics(np.random.default_rng(0).random(100), y)["auc"]
    assert 0.3 < r < 0.7


def test_lira_ranks_low_loss_as_member():
    sh = np.random.default_rng(0).normal(2.0, 0.1, size=(4, 10))
    target = np.full(10, 2.0)
    target[:5] = 0.5  # much lower loss than OUT -> member-like
    s = D.lira_offline_scores(target, sh)
    assert s[:5].min() > s[5:].max()


def test_dp_finetune_reports_accounting():
    torch.manual_seed(0)
    base = TinyLM(TINY)
    ds = D.build_dataset(qa.make_split(40, seed=1), D.make_canaries(4, 2), TINY.max_len)
    m, info = D.finetune(base, ds, 0, epochs=1, batch=8, lr=1e-3, epsilon=8.0, delta=1e-5, max_grad_norm=1.0)
    assert info["noise_multiplier"] > 0 and info["steps"] >= 1
    assert 0 < info["epsilon_spent_rdp"] <= 8.0 + 1e-6
    assert D.canary_losses(m, D.make_canaries(3, 3)).shape == (3,)


def test_fedavg_runs_and_accounts_comm():
    torch.manual_seed(0)
    base = TinyLM(TINY)
    data = {s: qa.make_site_split(8, i, s) for i, s in enumerate(["TOR", "MTL", "US"])}
    m, comm = fed_train(base, data, rounds=1, local_steps=1, batch=4, lr=1e-3, mu=0.01, seed=0)
    assert comm["bytes_per_round"] == 2 * 3 * base.n_params() * 4
    assert isinstance(m, TinyLM)


def test_clip_only_has_no_noise_and_no_epsilon():
    torch.manual_seed(0)
    base = TinyLM(TINY)
    ds = D.build_dataset(qa.make_split(24, seed=1), [], TINY.max_len)
    m, info = D.finetune(
        base, ds, 0, epochs=1, batch=8, lr=1e-3, epsilon=None, delta=1e-5, max_grad_norm=1.0, clip_only=True
    )
    assert info["noise_multiplier"] == 0.0 and info["clip_only"] and "epsilon_spent_rdp" not in info
    assert isinstance(m, TinyLM)
    changed = any((p0 - p1).abs().sum() > 0 for p0, p1 in zip(base.parameters(), m.parameters(), strict=True))
    assert changed
