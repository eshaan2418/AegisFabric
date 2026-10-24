"""Randomised correctness sweep of the fused kernels vs the fp64 PyTorch reference.

TRITON_INTERPRET=1 python -m bench.kernels_correctness --impl triton    # in aegis-kernels container
uv run python -m bench.kernels_correctness --impl metal                 # macOS host (Apple GPU)
uv run python -m bench.kernels_correctness --report                     # writes results/m7/correctness.md
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from bench.meta import RESULTS, write_result
from kernels import prefix_attn as K

TOL = {"fp32": 2e-5, "fp16": 2e-3}
DT = {"fp32": torch.float32, "fp16": torch.float16}
EDGE = [([1], [0]), ([1], [1]), ([7], [7]), ([7], [0]), ([33, 1, 2], [33, 0, 1]), ([200], [16])]


def cases(n_random: int, seed: int):
    rng = np.random.default_rng(seed)
    out = [(s, p, 2, 32) for s, p in EDGE]
    for _ in range(n_random):
        b = int(rng.integers(1, 7))
        seqlens = rng.integers(1, 201, size=b).tolist()
        pls = [int(rng.integers(0, s + 1)) for s in seqlens]
        out.append((seqlens, pls, int(rng.choice([1, 2, 4])), int(rng.choice([16, 32, 64]))))
    return out


def run(impl: str, n_random: int, seed: int) -> dict:
    if impl == "triton":
        fn = lambda q, k, v, cu, pl, L: K.triton_prefix_attn(q, k, v, cu, pl, L)  # noqa: E731
        dev = "cpu"
    else:
        from kernels.metal_prefix_attn import metal_prefix_attn

        fn = lambda q, k, v, cu, pl, L: metal_prefix_attn(q, k, v, cu, pl).cpu()  # noqa: E731
        dev = "mps"
    rows = []
    for ci, (seqlens, pls, h, d) in enumerate(cases(n_random, seed)):
        for dt in ("fp32", "fp16"):
            q, k, v, cu, pl = K.make_inputs(seqlens, pls, h, d, dtype=DT[dt], seed=ci)
            ref = K.reference_packed(q.double(), k.double(), v.double(), cu, pl)
            # context: error of the plain PyTorch reference evaluated in the same dtype
            base = K.reference_packed(q, k, v, cu, pl).double()
            out = fn(*(t.to(dev) for t in (q, k, v)), cu, pl, max(seqlens)).double()
            err = (out - ref).abs()
            rows.append(
                {
                    "case": ci,
                    "seqlens": seqlens,
                    "prefix": pls,
                    "H": h,
                    "D": d,
                    "dtype": dt,
                    "max_abs_err": float(err.max()),
                    "mean_abs_err": float(err.mean()),
                    "torch_same_dtype_max_err": float((base - ref).abs().max()),
                    "pass": bool(err.max() <= TOL[dt]),
                    "finite": bool(torch.isfinite(out).all()),
                }
            )
    n_pass = sum(r["pass"] and r["finite"] for r in rows)
    return {
        "impl": impl,
        "tolerances": TOL,
        "n": len(rows),
        "n_pass": n_pass,
        "rows": rows,
        "triton_interpret": impl == "triton",
    }


def report() -> None:
    lines = [
        "# M7 correctness report",
        "",
        "Reference: `kernels.prefix_attn.reference_packed` in fp64. Inputs ~N(0,1). Tolerance = max |err| over",
        "all outputs: fp32 ≤ 2e-5, fp16 ≤ 2e-3. 'torch same-dtype' is the error of the PyTorch reference evaluated",
        "in the input dtype (fp16 inputs are computed in fp32 by the reference), given for context.",
        "",
    ]
    for impl in ("triton", "metal"):
        p = RESULTS / "m7" / f"correctness_{impl}.json"
        if not p.exists():
            lines.append(f"## {impl}: not run\n")
            continue
        d = json.loads(p.read_text())
        data, meta = d["data"], d["meta"]
        lines += [
            f"## {impl} ({'TRITON_INTERPRET=1, Linux container' if impl == 'triton' else 'Apple GPU (MPS)'})",
            "",
            f"- result file: `results/m7/correctness_{impl}.json` (commit `{meta['git']['commit'][:10]}`)",
            f"- **{data['n_pass']}/{data['n']} cases pass**",
            "",
        ]
        lines += [
            "| dtype | cases | worst max-abs-err | median max-abs-err | worst torch same-dtype |",
            "|---|---|---|---|---|",
        ]
        for dt in ("fp32", "fp16"):
            rs = [r for r in data["rows"] if r["dtype"] == dt]
            e = [r["max_abs_err"] for r in rs]
            lines.append(
                f"| {dt} | {len(rs)} | {max(e):.2e} | {np.median(e):.2e} | "
                f"{max(r['torch_same_dtype_max_err'] for r in rs):.2e} |"
            )
        lines.append("")
    (RESULTS / "m7" / "correctness.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["triton", "metal"])
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.impl:
        res = run(a.impl, a.n_random, a.seed)
        write_result(f"m7/correctness_{a.impl}.json", res, config=vars(a), seed=a.seed)
        print(f"{a.impl}: {res['n_pass']}/{res['n']} pass")
    if a.report:
        report()


if __name__ == "__main__":
    main()
