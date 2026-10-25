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
