import numpy as np

from bench import stats


def test_summary_ci_contains_mean():
    s = stats.summarize([1.0, 2.0, 3.0, 4.0, 5.0], with_p99=True)
    assert s.n == 5 and s.mean == 3.0
    assert s.ci95_lo < 3.0 < s.ci95_hi
    assert abs((s.ci95_hi - s.ci95_lo) / 2 - 2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)) < 1e-2


def test_welch_detects_difference():
    rng = np.random.default_rng(0)
    r = stats.welch(rng.normal(10, 1, 30), rng.normal(12, 1, 30))
    assert r["p_welch"] < 1e-6 and r["hedges_g"] < 0


def test_holm_monotone():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
