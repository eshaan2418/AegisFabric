"""Latency / effective-bandwidth benchmark for varlen prefix-LM attention.

CPU implementations (run in the Linux lab container, pinned to 4 cores):
    eager (padded unfused), torch.compile(inductor) of the same function, SDPA(bool mask), JAX jit.
MPS implementations (run on the macOS host, Apple GPU):
    eager (padded unfused), SDPA(bool mask), custom fused Metal kernel (packed varlen).
The Triton kernel cannot be timed here: no CUDA GPU; its interpreter mode is a numpy emulator (correctness only).

    python -m bench.kernels_perf --device cpu --threads 4     # inside aegis-kernels, --cpuset-cpus=0-3
    uv run python -m bench.kernels_perf --device mps          # host
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from bench.meta import write_result
from bench.stats import summarize
from kernels import prefix_attn as K

SHAPES = {
    # name: (B, seqlen range, prefix, H, D)
    "model": (16, (64, 256), 16, 4, 32),
    "large": (32, (256, 1024), 64, 8, 64),
}


def make_case(name: str, dtype, seed=0):
    b, (lo, hi), pre, h, d = SHAPES[name]
    rng = np.random.default_rng(seed)
    seqlens = rng.integers(lo, hi + 1, size=b).tolist()
    pls = [min(pre, s) for s in seqlens]
    return seqlens, pls, h, d, K.make_inputs(seqlens, pls, h, d, dtype=dtype, seed=seed)


def traffic_model(seqlens, pls, h, d, elt: int, bm: int = 32) -> dict:
    """Bytes moved to/from device memory under three models."""
    t = sum(seqlens)
    L = max(seqlens)
    b = len(seqlens)
    fused_min = 4 * t * h * d * elt  # read q,k,v once, write o
    # tiled fused (this kernel): each query block re-reads the K/V rows it can see (upper bound: all rows <= hi)
    kv_rows = sum(
        min(n, max(p, m0 + bm)) for n, p in zip(seqlens, pls, strict=True) for m0 in range(0, n, bm)
    )
    fused_tiled = (2 * t + 2 * kv_rows) * h * d * elt
    # unfused padded lower bound: padded qkv/o + write S, read S/write P, read P (fp32 scores)
    unfused = 4 * b * L * h * d * elt + 3 * b * h * L * L * 4 + b * L * L
    return {
        "fused_min": fused_min,
        "fused_tiled": fused_tiled,
        "unfused_padded_lb": unfused,
        "padding_waste": 1 - t / (b * L),
    }


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def time_fn(fn, dev, trials: int, iters: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(dev)
    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(dev)
        out.append((time.perf_counter() - t0) / iters * 1e3)  # ms
    return out


def peak_copy_bw(dev, trials: int) -> dict:
    n = 128 * 2**20  # 128M floats = 512 MiB
    src = torch.ones(n, device=dev)
    dst = torch.empty_like(src)
    ts = time_fn(lambda: dst.copy_(src), dev, trials, iters=3)
    bw = [2 * n * 4 / (t / 1e3) / 1e9 for t in ts]
    return {"bytes": 2 * n * 4, "GBps": summarize(bw).as_dict()}


def impls_for(dev, case, dtype):
    seqlens, pls, h, d, (q, k, v, cu, pl) = case
    L = max(seqlens)
    scale = 1 / math.sqrt(d)
    qp, kp, vp = (K.pack_to_padded(t, cu, L).to(dev) for t in (q, k, v))
    mask = K.padded_mask(torch.tensor(seqlens), pl, L).to(dev)
    ref = K.reference_packed(q.double(), k.double(), v.double(), cu, pl)
    impls = {
        "eager_unfused": lambda: K.padded_unfused(qp, kp, vp, mask, scale),
        "sdpa_mask": lambda: torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask),
    }
    packed = set()  # impls whose output is already in packed (total, H, D) layout
    if dev == "cpu":
        comp = torch.compile(K.padded_unfused, dynamic=False)
        impls["torch_compile"] = lambda: comp(qp, kp, vp, mask, scale)
        try:
            import jax
            import jax.numpy as jnp

            def jfn(q_, k_, v_, m_):
                s = jnp.einsum("bhqd,bhkd->bhqk", q_, k_) * scale
                s = jnp.where(m_, s, -jnp.inf)
                return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), v_)

            jj = jax.jit(jfn)
            jargs = [jnp.asarray(t.cpu().numpy()) for t in (qp, kp, vp, mask)]
            impls["jax_jit"] = lambda: jj(*jargs).block_until_ready()
        except Exception as e:  # pragma: no cover
            print("jax unavailable:", e)
    if dev == "mps":
        from kernels.metal_prefix_attn import block_table, metal_prefix_attn

        qm, km, vm = (t.to("mps") for t in (q, k, v))
        cum, plm = cu.to("mps"), pl.to("mps")
        blk = block_table(cu, 32).to("mps")
        impls["metal_fused"] = lambda: metal_prefix_attn(qm, km, vm, cum, plm, blk=blk)
        packed.add("metal_fused")

    def check(name, fn):
        o = fn()
        if name == "jax_jit":
            o = torch.from_numpy(np.asarray(o))
        o = o.cpu()
        if name not in packed:
            o = K.padded_to_pack(o, cu)
        return float((o.double() - ref).abs().max())

    return impls, check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--shapes", default="model,large")
    ap.add_argument("--dtypes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = a.device
    dtypes = a.dtypes.split(",") if a.dtypes else (["fp32"] if dev == "cpu" else ["fp32", "fp16"])
    dmap = {"fp32": torch.float32, "fp16": torch.float16}
    res: dict = {
        "device": dev,
        "threads": a.threads,
        "note": "dev run, machine shared",
        "peak_copy": peak_copy_bw(dev, a.trials),
        "cases": [],
    }
    peak = res["peak_copy"]["GBps"]["mean"]
    print(f"peak copy {peak:.1f} GB/s")
    for shape in a.shapes.split(","):
        for dt in dtypes:
            dtype = dmap[dt]
            case = make_case(shape, dtype, a.seed)
            seqlens, pls, h, d, _ = case
            tm = traffic_model(seqlens, pls, h, d, torch.finfo(dtype).bits // 8)
            impls, check = impls_for(dev, case, dtype)
            iters = 2 if shape == "large" else 10
            for name, fn in impls.items():
                err = check(name, fn)
                ts = time_fn(fn, dev, a.trials, iters)
                s = summarize(ts)
                eff = tm["fused_min"] / (s.mean / 1e3) / 1e9
                rec = {
                    "shape": shape,
                    "dtype": dt,
                    "impl": name,
                    "latency_ms": s.as_dict(),
                    "max_abs_err": err,
                    "eff_GBps_fused_min": eff,
                    "frac_of_peak": eff / peak,
                    "traffic": tm,
                    "seqlens_total": sum(seqlens),
                    "max_len": max(seqlens),
                }
                res["cases"].append(rec)
                print(f"{shape:6s} {dt} {name:14s} {s.fmt('ms')}  err={err:.2e} eff={eff:.2f}GB/s")
    write_result(f"m7/perf_{dev}.json", res, config=vars(a), seed=a.seed)


if __name__ == "__main__":
    main()
