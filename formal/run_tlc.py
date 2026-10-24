"""Run TLC on the models in formal/models.py and summarise results.

Usage: uv run python -m formal.run_tlc [--small] [--only NAME ...]
Raw TLC output (including counterexample traces) goes to results/m9/<name>.out; the summary to
results/m9/summary.json (with provenance via bench.meta.write_result).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench.meta import RESULTS, write_result
from formal.models import FORMAL, MODELS, Model

JAR = FORMAL / "tools" / "tla2tools.jar"
OUT = RESULTS / "m9"


def find_java() -> str | None:
    for cand in (os.environ.get("JAVA"), "/opt/homebrew/opt/openjdk/bin/java", shutil.which("java")):
        if not cand or not Path(cand).exists():
            continue
        if subprocess.run([cand, "-version"], capture_output=True).returncode == 0:
            return cand
    return None


def parse(out: str) -> dict:
    def grab(pat, cast=int):
        ms = re.findall(pat, out)
        return cast(ms[-1].replace(",", "")) if ms else None

    violated = re.search(
        r"Error: (Invariant (\S+) is violated|Temporal propert(?:y|ies) (\S+ )?(?:was|were) violated)", out
    )
    error = re.search(r"^Error: (.*)$", out, re.M)
    return {
        "states_generated": grab(r"([\d,]+) states generated"),
        "distinct_states": grab(r"([\d,]+) distinct states found"),
        "depth": grab(r"depth of the complete state graph search is (\d+)"),
        "violation": (violated.group(2) or (violated.group(3) or "temporal").strip()) if violated else None,
        "error": error.group(1) if error and not violated else None,
        "trace_len": len(re.findall(r"^State \d+:", out, re.M)),
        "completed": "Model checking completed. No error has been found." in out,
    }


def run(model: Model, java: str, workers: int, out_dir: Path = OUT) -> dict:
    with tempfile.TemporaryDirectory() as meta:
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                java,
                "-XX:+UseParallelGC",
                "-cp",
                str(JAR),
                "tlc2.TLC",
                "-workers",
                str(workers),
                "-metadir",
                meta,
                "-config",
                model.cfg.name,
                "MC.tla",
            ],
            cwd=FORMAL / model.dir,
            capture_output=True,
            text=True,
        )
        wall = time.monotonic() - t0
    for f in (FORMAL / model.dir).glob("MC_TTrace_*.tla"):
        f.unlink()
    out = proc.stdout + proc.stderr
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model.name}.out").write_text(out)
    r = parse(out)
    outcome = "violation" if r["violation"] else ("pass" if r["completed"] else "error")
    r.update(
        name=model.name,
        spec=model.dir,
        expect=model.expect,
        outcome=outcome,
        as_expected=outcome == model.expect,
        wall_s=round(wall, 2),
        note=model.note,
        constants=model.constants,
        invariants=model.invariants,
        properties=model.properties,
    )
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="only models marked small")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-write", action="store_true", help="do not write summary.json")
    a = ap.parse_args()
    java = find_java()
    if not java or not JAR.exists():
        print("java or formal/tools/tla2tools.jar missing; run formal/get_tools.sh", file=sys.stderr)
        return 2
    models = [m for m in MODELS if (not a.small or m.small) and (not a.only or m.name in a.only)]
    for m in models:
        m.cfg.write_text(m.render())
    results = []
    for m in models:
        r = run(m, java, a.workers)
        results.append(r)
        print(
            f"{m.name:32s} {r['outcome']:9s} expect={m.expect:9s} states={r['distinct_states']} "
            f"depth={r['depth']} viol={r['violation']} trace={r['trace_len']} {r['wall_s']}s"
        )
    if not a.no_write:
        jar_sha = hashlib.sha256(JAR.read_bytes()).hexdigest()
        write_result(
            OUT / "summary.json",
            results,
            config={"tla2tools_sha256": jar_sha, "workers": a.workers, "small_only": a.small},
        )
    return 0 if all(r["as_expected"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
