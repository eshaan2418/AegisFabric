"""Compile the Triton prefix-attention kernel for a CUDA target *without a GPU* and dump the IR stages.

Run inside aegis-kernels (no TRITON_INTERPRET):  python -m bench.kernels_ir --out results/m7/ir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bench.meta import write_result


def compile_stages(arch: int, d: int, block_m: int, block_n: int, dtype: str, num_warps: int, hints: bool):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from kernels.prefix_attn import _prefix_attn_fwd

    ptr = f"*{dtype}"
    sig = {
        "Q": ptr,
        "K": ptr,
        "V": ptr,
        "O": ptr,
        "LSE": "*fp32",
        "CU": "*i32",
        "PL": "*i32",
        "stride_t": "i32",
        "stride_h": "i32",
        "scale": "fp32",
        "H": "constexpr",
        "D": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
    }
    # hints=True mirrors JIT launch-time specialisation: pointers 16-B aligned and strides divisible by 16
    # (true for contiguous (T,H,D) tensors with H*D % 16 == 0), which lets Triton emit vector loads.
    attrs = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3, 4, 7, 8)} if hints else None
    src = ASTSource(
        fn=_prefix_attn_fwd,
        signature=sig,
        attrs=attrs,
        constexprs={"H": 4, "D": d, "BLOCK_M": block_m, "BLOCK_N": block_n},
    )
    stages: dict = {}
    err = None
    meta: dict = {}
    try:
        k = triton.compile(src, target=GPUTarget("cuda", arch, 32), options={"num_warps": num_warps})
        stages = dict(k.asm)
        meta = {"shared_bytes": getattr(k.metadata, "shared", None), "num_warps": k.metadata.num_warps}
    except Exception as e:  # cubin needs ptxas; earlier stages may still be obtainable
        err = f"{type(e).__name__}: {str(e)[:600]}"
    return stages, meta, err


def ptx_stats(ptx: str) -> dict:
    ops: dict = {
        "ld.global": len(re.findall(r"\bld\.global", ptx)),
        "st.global": len(re.findall(r"\bst\.global", ptx)),
        "ld.shared": len(re.findall(r"\bld\.shared", ptx)),
        "st.shared": len(re.findall(r"\bst\.shared", ptx)),
        "cp.async": len(re.findall(r"\bcp\.async\.\w", ptx)),
        "ldmatrix": len(re.findall(r"\bldmatrix", ptx)),
        "mma": len(re.findall(r"\bmma\.sync", ptx)),
        "ex2": len(re.findall(r"\bex2\.approx", ptx)),
        "bar.sync": len(re.findall(r"\bbar\.sync|barrier\.sync", ptx)),
    }
    vec = re.findall(r"ld\.global(?:\.\w+)*?\.v(\d)\.(?:b|f|u)(\d+)", ptx)
    ops["ld.global vector widths"] = sorted({f"v{a}x{b}" for a, b in vec})
    return ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m7/ir")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import triton

    summary: dict = {"triton": triton.__version__, "configs": []}
    for arch, dtype, hints in [(80, "fp16", False), (80, "fp16", True), (80, "fp32", True)]:
        cfg = dict(arch=arch, d=64, block_m=64, block_n=32, dtype=dtype, num_warps=4, hints=hints)
        stages, meta, err = compile_stages(**cfg)
        tag = f"sm{arch}_{dtype}_d64_m64_n32" + ("_div16" if hints else "_nohints")
        for name, text in stages.items():
            if name in ("ttir", "ttgir", "ptx") and isinstance(text, str):
                (out / f"{tag}.{name}").write_text(text)
        rec: dict = {"config": cfg, "stages": sorted(stages), "metadata": meta, "error": err}
        if "ptx" in stages:
            rec["ptx_stats"] = ptx_stats(stages["ptx"])
        if "ttgir" in stages:
            rec["ttgir_local_allocs"] = len(re.findall(r"ttg\.local_alloc", stages["ttgir"]))
        summary["configs"].append(rec)
        print(tag, sorted(stages), err)
    write_result("m7/ir_summary.json", summary)


if __name__ == "__main__":
    main()
