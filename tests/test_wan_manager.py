from pathlib import Path
from types import SimpleNamespace

from bench.wan_run import JobManager


def _mgr(tmp_path: Path, storage: str) -> JobManager:
    cfg = SimpleNamespace(scenario="sc", label="joint", seed=0, ckpt_backend="blob")
    m = JobManager.__new__(JobManager)
    m.cfg, m.out = cfg, tmp_path
    m.plan = SimpleNamespace(placement=("tor",) * 3, storage=storage)
    return m


def _put(root: Path, step: int, ranks, world: int = 3) -> None:
    root.mkdir(exist_ok=True)
    for r in ranks:
        (root / f"ckpt_sc-joint-0_{step:08d}_slice{r}of{world}").write_bytes(b"x")


def test_newest_complete_checkpoint_skips_partial_and_spans_sites(tmp_path):
    _put(tmp_path / "store_tor", 20, range(3))
    _put(tmp_path / "store_tor", 40, [0, 1])  # incomplete
    _put(tmp_path / "store_mtl", 30, range(3))  # newer, at the other (pre-migration) site
    _put(tmp_path / "store_mtl", 60, range(3), world=6)  # different world size: ignored
    (tmp_path / "store_tor" / ".ckpt_sc-joint-0_00000040_slice2of3.tmp").write_bytes(b"")
    assert _mgr(tmp_path, "tor").newest_complete_checkpoint() == (30, "mtl")


def test_newest_complete_checkpoint_none(tmp_path):
    assert _mgr(tmp_path, "tor").newest_complete_checkpoint() == (-1, None)


def test_stable_gates_migration_until_segment_runs(tmp_path, monkeypatch):
    import time

    import bench.wan_run as W

    m = _mgr(tmp_path, "tor")
    m.segment, m.events = 0, []
    monkeypatch.setattr(W.JobManager, "alive", lambda self: True)
    seg = tmp_path / "seg0"
    seg.mkdir()
    monkeypatch.setattr(W, "read_rank_log", lambda d, r: [{"event": "step"}] * n_steps[0])
    n_steps = [2]
    assert not m.stable()  # still starting / restoring
    n_steps[0] = 10
    assert m.stable()
    m.events.append({"event": "migrate", "wall": time.time()})
    assert not m.stable()  # cooldown
