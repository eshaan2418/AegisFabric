"""Audit: every number in the write-up documents must be backed by a value in results/.

Scans README.md, RESUME_BULLETS.md, FINAL_REVIEW.md and paper/*.tex. For each numeric token, it looks for a
value anywhere in results/**/*.json (data *and* recorded config/meta) or results/**/*.md tables that matches
at the precision shown (e.g. "12.3" matches 12.27 or 12.34; "31%" matches 0.31 or 31.x). Numbers that are
structural (years, section/figure/milestone indices, citation keys, versions, list ordinals) are skipped by
pattern; anything else unmatched is reported. Exit 1 if any unbacked number remains.

    python -m bench.audit_numbers [--verbose]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from bench.meta import REPO_ROOT

DOCS = ["README.md", "RESUME_BULLETS.md", "FINAL_REVIEW.md"]
NUM = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![\w.])\d+(?:\.\d+)?(?!\.?\d)")
SKIP_CONTEXT = re.compile(
    r"(M\d+|P\d|I\d|R\d|D-\d+|B-\d+|§|Fig\.?|Figure|Table|Section|Sec\.|seed\s*\d|v\d|\d{4}-\d{2}-\d{2}|"
    r"TLS\s*1\.3|AES-256|SHA-?256|P-256|Ed25519|BLAKE3|X\.509|IPv4|sm_?80|fp(16|32|64)|bf16|"
    r"\\cite\{[^}]*|\\ref\{[^}]*|\\label\{[^}]*|https?://\S+|`[^`]*`)"
)


def collect_values() -> list[float]:
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            if math.isfinite(o):
                vals.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for m in re.findall(r"[-+]?\d+(?:\.\d+)?", o):
                vals.append(float(m))

    for p in (REPO_ROOT / "results").rglob("*.json"):
        try:
            walk(json.loads(p.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    for p in (REPO_ROOT / "results").rglob("*.md"):
        for m in re.findall(r"[-+]?\d+(?:\.\d+)?", p.read_text(errors="ignore")):
            vals.append(float(m))
    return vals


def matches(token: str, vals_sorted: list[float], pct: bool) -> bool:
    t = token.replace(",", "")
    x = float(t)
    dec = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    cands = [x, -x]
    if pct:
        cands += [x / 100, -x / 100]
    import bisect

    for c in cands:
        ctol = tol / 100 if pct and abs(c) == abs(x) / 100 else tol
        i = bisect.bisect_left(vals_sorted, c - ctol)
        if i < len(vals_sorted) and vals_sorted[i] <= c + ctol:
            return True
    return False


def strip_skips(line: str) -> str:
    return SKIP_CONTEXT.sub(" ", line)


def audit(paths: list[Path], verbose: bool = False) -> list[str]:
    vals = sorted(collect_values())
    bad = []
    for p in paths:
        if not p.exists():
            continue
        in_code = False
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.strip().startswith(("%", "#!", "<!--")):
                continue
            text = strip_skips(line)
            for m in NUM.finditer(text):
                tok = m.group(0)
                bare = tok.replace(",", "").lstrip("+-")
                if bare in ("0", "1", "2", "3") or re.fullmatch(r"(19|20)\d\d", bare):
                    continue  # trivial small integers / years
                after = text[m.end() : m.end() + 2]
                pct = after.startswith("%") or after.startswith(r"\%")
                if not matches(tok, vals, pct):
                    bad.append(f"{p.relative_to(REPO_ROOT)}:{ln}: {tok}   | {line.strip()[:120]}")
                elif verbose:
                    print(f"ok {p.name}:{ln} {tok}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    paths = [REPO_ROOT / d for d in DOCS] + sorted((REPO_ROOT / "paper").glob("*.tex"))
    bad = audit(paths, a.verbose)
    if bad:
        print(f"audit_numbers: {len(bad)} unbacked number(s):")
        print("\n".join(bad))
        return 1
    print(
        f"audit_numbers: all numbers in {len([p for p in paths if p.exists()])} documents are backed by results/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
