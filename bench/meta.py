"""Result provenance: every result file written by this repo goes through `write_result`.

Each file records git commit (and dirty flag), hardware, seed, config, command line and
timestamp, so that any number quoted in docs can be traced to the run that produced it.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return os.environ.get("AEGIS_GIT_COMMIT", "unknown")


def git_info() -> dict[str, Any]:
    commit = os.environ.get("AEGIS_GIT_COMMIT") or _git("rev-parse", "HEAD")
    dirty_env = os.environ.get("AEGIS_GIT_DIRTY")
    if dirty_env is not None:
        dirty = dirty_env == "1"
    else:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "host_label": os.environ.get("AEGIS_HOST_LABEL", platform.system().lower()),
    }
    try:
        import psutil

        info["mem_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return info


def metadata(config: dict[str, Any] | None = None, seed: int | None = None) -> dict[str, Any]:
    return {
        "git": git_info(),
        "hardware": hardware_info(),
        "seed": seed,
        "config": config or {},
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else "",
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


def write_result(
    path: str | Path, data: Any, config: dict[str, Any] | None = None, seed: int | None = None
) -> Path:
    """Write `data` with provenance metadata to `path` (JSON). Returns the path."""
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS / p
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": metadata(config, seed), "data": data}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    os.replace(tmp, p)
    return p


def read_result(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS / p
    return json.loads(p.read_text())


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
