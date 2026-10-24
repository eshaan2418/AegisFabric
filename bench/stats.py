"""Statistics helpers used by every experiment: mean ± 95% CI, tests, multiple-comparison correction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as _st


@dataclass
class Summary:
    n: int
    mean: float
    std: float
    ci95_lo: float
    ci95_hi: float
    median: float
    p99: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def fmt(self, unit: str = "", digits: int = 3) -> str:
        half = (self.ci95_hi - self.ci95_lo) / 2
        return f"{self.mean:.{digits}g} ± {half:.{digits}g}{unit} (n={self.n})"


def summarize(xs: Sequence[float], with_p99: bool = False) -> Summary:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan, None)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(_st.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    else:
        half = float("nan")
    p99 = float(np.percentile(a, 99)) if with_p99 else None
    return Summary(n, mean, std, mean - half, mean + half, float(np.median(a)), p99)


def welch(a: Sequence[float], b: Sequence[float]) -> dict:
    """Welch's t-test (two-sided) plus Mann-Whitney U and Cliff's-delta-free effect size (Hedges g)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    t = _st.ttest_ind(a, b, equal_var=False)
    try:
        u = _st.mannwhitneyu(a, b, alternative="two-sided")
        u_p = float(u.pvalue)
    except ValueError:
        u_p = float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    g = float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")
    g *= 1 - 3 / (4 * (na + nb) - 9) if na + nb > 3 else 1.0
    return {
        "t": float(t.statistic),
        "p_welch": float(t.pvalue),
        "p_mannwhitney": u_p,
        "hedges_g": g,
        "rel_diff": float((a.mean() - b.mean()) / b.mean()) if b.mean() != 0 else float("nan"),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def bootstrap_ci(xs: Sequence[float], stat=np.mean, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    bs = np.apply_along_axis(stat, 1, a[idx])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
