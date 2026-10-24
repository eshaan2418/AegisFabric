"""Fused varlen prefix-LM attention as a custom Metal kernel (Apple GPU), via torch.mps.compile_shader.

Same algorithm as the Triton kernel in `prefix_attn.py`: one threadgroup per (query block, head). Each thread
owns one query row (q and the output accumulator stay in registers). K/V tiles of BLOCK_N keys are staged
cooperatively in threadgroup memory, and an online softmax means the score matrix never reaches device memory.
Query blocks never span sequences: the host builds a block table (seq_id, q_start), as varlen FlashAttention does.
"""

from __future__ import annotations

import math

import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#define D {D}
#define BM {BM}
#define BN {BN}
typedef {T} T;

kernel void prefix_attn(
    device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], device const T* V [[buffer(2)]],
    device T* O [[buffer(3)]], device const int* CU [[buffer(4)]], device const int* PL [[buffer(5)]],
    device const int* BLK [[buffer(6)]], constant int& H [[buffer(7)]], constant float& scale [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint tid = tid3.x;
    const int blk = tg.x, h = tg.y;
    const int b = BLK[2 * blk], m0 = BLK[2 * blk + 1];
    const int start = CU[b], n = CU[b + 1] - start, plen = PL[b];
    const int i = m0 + int(tid);           // query row within sequence
    const bool row_ok = i < n;
    const int stride_t = H * D;
    threadgroup float ks[BN][D];
    threadgroup float vs[BN][D];
    float q[D], acc[D];
    for (int d = 0; d < D; ++d) { q[d] = row_ok ? float(Q[(start + i) * stride_t + h * D + d]) * scale : 0.0f; acc[d] = 0.0f; }
    float m_i = -INFINITY, l_i = 0.0f;
    const int hi = min(n, max(plen, m0 + BM));
    for (int n0 = 0; n0 < hi; n0 += BN) {
        // cooperative tile load: BM threads load BN*D elements of K and V
        for (int e = int(tid); e < BN * D; e += BM) {
            int r = e / D, d = e % D, j = n0 + r;
            bool ok = j < n;
            ks[r][d] = ok ? float(K[(start + j) * stride_t + h * D + d]) : 0.0f;
            vs[r][d] = ok ? float(V[(start + j) * stride_t + h * D + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[BN];
        float m_blk = -INFINITY;
        for (int r = 0; r < BN; ++r) {
            int j = n0 + r;
            bool vis = (j < n) && (j < plen || j <= i);
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += q[d] * ks[r][d];
            s[r] = vis ? dot : -INFINITY;
            m_blk = max(m_blk, s[r]);
        }
        float m_new = max(m_i, m_blk);
        if (m_new > -INFINITY) {
            float alpha = exp(m_i - m_new);
            l_i *= alpha;
            for (int d = 0; d < D; ++d) acc[d] *= alpha;
            for (int r = 0; r < BN; ++r) {
                float p = exp(s[r] - m_new);
                l_i += p;
                for (int d = 0; d < D; ++d) acc[d] += p * vs[r][d];
            }
            m_i = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row_ok) {
        float inv = 1.0f / l_i;
        for (int d = 0; d < D; ++d) O[(start + i) * stride_t + h * D + d] = T(acc[d] * inv);
    }
}
"""

_CACHE: dict = {}


def _lib(d: int, bm: int, bn: int, dtype: torch.dtype):
    key = (d, bm, bn, dtype)
    if key not in _CACHE:
        t = {torch.float32: "float", torch.float16: "half"}[dtype]
        _CACHE[key] = torch.mps.compile_shader(
            _SRC.replace("{D}", str(d)).replace("{BM}", str(bm)).replace("{BN}", str(bn)).replace("{T}", t)
        )
    return _CACHE[key]


def block_table(cu_seqlens: torch.Tensor, bm: int) -> torch.Tensor:
    cu = cu_seqlens.tolist()
    rows = [(b, m0) for b in range(len(cu) - 1) for m0 in range(0, cu[b + 1] - cu[b], bm)]
    return torch.tensor(rows, dtype=torch.int32).reshape(-1)


def metal_prefix_attn(
    q,
    k,
    v,
    cu_seqlens,
    prefix_lens,
    scale: float | None = None,
    bm: int = 32,
    bn: int = 16,
    blk: torch.Tensor | None = None,
):
    """q,k,v: (total, H, D) contiguous on 'mps'. cu_seqlens/prefix_lens int32 (any device)."""
    total, h, d = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    lib = _lib(d, bm, bn, q.dtype)
    if blk is None:
        blk = block_table(cu_seqlens.cpu(), bm).to("mps")
    o = torch.empty_like(q)
    nblk = blk.numel() // 2
    lib.prefix_attn(
        q,
        k,
        v,
        o,
        cu_seqlens.to("mps", torch.int32),
        prefix_lens.to("mps", torch.int32),
        blk,
        h,
        float(scale),
        threads=(nblk * bm, h, 1),
        group_size=(bm, 1, 1),
    )
    return o
