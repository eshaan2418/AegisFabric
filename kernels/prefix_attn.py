"""Variable-length prefix-LM attention over packed sequences.

Packed layout: q, k, v have shape (total_tokens, H, D); sequence b occupies rows
cu_seqlens[b]:cu_seqlens[b+1]. Within a sequence, query i may attend to key j iff
    j < prefix_len[b]   (image-patch prefix: bidirectional)   or   j <= i   (text: causal).
No attention crosses sequence boundaries.

Implementations:
  * reference_packed  — PyTorch, per-sequence loop, explicit mask (ground truth, fp64-capable).
  * padded_unfused    — PyTorch, padded batch, materialises the (B,H,L,L) score matrix (the "eager" baseline;
                        also the function handed to torch.compile and mirrored in JAX).
  * triton_prefix_attn — fused flash-style Triton kernel (online softmax, no score matrix in memory).
"""

from __future__ import annotations

import math

import torch


def visible(i: torch.Tensor, j: torch.Tensor, prefix_len: int) -> torch.Tensor:
    return (j < prefix_len) | (j <= i)


def reference_packed(q, k, v, cu_seqlens, prefix_lens, scale: float | None = None) -> torch.Tensor:
    """Ground-truth prefix-LM attention. Computes in fp64 when inputs are fp64, else fp32; returns input dtype."""
    d = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.empty_like(q)
    cu = cu_seqlens.tolist()
    pl = prefix_lens.tolist()
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        n = e - s
        if n == 0:
            continue
        qb, kb, vb = (t[s:e].to(acc_dtype).transpose(0, 1) for t in (q, k, v))  # (H, n, D)
        scores = qb @ kb.transpose(-1, -2) * scale
        idx = torch.arange(n)
        mask = visible(idx[:, None], idx[None, :], pl[b])
        scores = scores.masked_fill(~mask, float("-inf"))
        out[s:e] = (torch.softmax(scores, -1) @ vb).transpose(0, 1).to(q.dtype)
    return out


def pack_to_padded(x: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int) -> torch.Tensor:
    """(total, H, D) → (B, H, L, D) zero-padded."""
    b = len(cu_seqlens) - 1
    h, d = x.shape[1], x.shape[2]
    out = x.new_zeros(b, max_len, h, d)
    cu = cu_seqlens.tolist()
    for i in range(b):
        out[i, : cu[i + 1] - cu[i]] = x[cu[i] : cu[i + 1]]
    return out.transpose(1, 2).contiguous()


def padded_to_pack(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    cu = cu_seqlens.tolist()
    xt = x.transpose(1, 2)
    return torch.cat([xt[i, : cu[i + 1] - cu[i]] for i in range(len(cu) - 1)], 0)


def padded_mask(seqlens: torch.Tensor, prefix_lens: torch.Tensor, max_len: int, device=None) -> torch.Tensor:
    """Boolean (B, 1, L, L) mask, True = attend. Padded query rows attend to key 0 to stay finite."""
    idx = torch.arange(max_len, device=device)
    i, j = idx[None, :, None], idx[None, None, :]
    valid_k = j < seqlens.to(device)[:, None, None]
    m = (visible(i, j, 0) | (j < prefix_lens.to(device)[:, None, None])) & valid_k
    m = m | (j == 0)
    return m[:, None]


def padded_unfused(q, k, v, mask, scale: float):
    """Unfused baseline: S = QK^T (materialised), masked softmax, PV. q,k,v: (B,H,L,D)."""
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    p = torch.softmax(s.to(acc), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


# ----------------------------------------------------------------------------- Triton

try:  # Triton is optional (Linux only); the module stays importable without it.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - platform dependent
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _prefix_attn_fwd(
        Q,
        K,
        V,
        O,  # noqa: E741
        LSE,
        CU,
        PL,
        stride_t,
        stride_h,  # token / head strides (elements); last dim contiguous
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        start = tl.load(CU + b)
        n = tl.load(CU + b + 1) - start
        plen = tl.load(PL + b)
        m0 = pid_m * BLOCK_M
        if m0 < n:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, D)
            row_ok = offs_m < n
            q_ptrs = Q + (start + offs_m)[:, None] * stride_t + h * stride_h + offs_d[None, :]
            q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
            m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
            l_i = tl.zeros([BLOCK_M], tl.float32)
            acc = tl.zeros([BLOCK_M, D], tl.float32)
            # Keys beyond max(prefix, last query in block) are never visible: skip those blocks entirely.
            hi = tl.minimum(n, tl.maximum(plen, m0 + BLOCK_M))
            for n0 in range(0, hi, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                col_ok = offs_n < n
                kv_off = (start + offs_n)[:, None] * stride_t + h * stride_h + offs_d[None, :]
                k = tl.load(K + kv_off, mask=col_ok[:, None], other=0.0)
                v = tl.load(V + kv_off, mask=col_ok[:, None], other=0.0)
                s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
                vis = ((offs_n[None, :] < plen) | (offs_n[None, :] <= offs_m[:, None])) & col_ok[None, :]
                s = tl.where(vis, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
            acc = acc / l_i[:, None]
            o_ptrs = O + (start + offs_m)[:, None] * stride_t + h * stride_h + offs_d[None, :]
            tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=row_ok[:, None])
            tl.store(LSE + (start + offs_m) * H + h, m_i + tl.log(l_i), mask=row_ok)


def triton_prefix_attn(
    q,
    k,
    v,
    cu_seqlens,
    prefix_lens,
    max_seqlen: int,
    scale: float | None = None,
    block_m: int = 32,
    block_n: int = 32,
    return_lse: bool = False,
):
    if not HAVE_TRITON:
        raise RuntimeError("triton not available")
    total, h, d = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert d in (16, 32, 64, 128), "head dim must be a power of two >= 16 (tl.dot constraint)"
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    o = torch.empty_like(q)
    lse = torch.empty(total, h, dtype=torch.float32, device=q.device)
    b = len(cu_seqlens) - 1
    grid = (triton.cdiv(max_seqlen, block_m), b * h)
    _prefix_attn_fwd[grid](
        q,
        k,
        v,
        o,
        lse,
        cu_seqlens.to(torch.int32),
        prefix_lens.to(torch.int32),
        q.stride(0),
        q.stride(1),
        scale,
        H=h,
        D=d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return (o, lse) if return_lse else o


def make_inputs(seqlens, prefix_lens, h, d, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    total = int(sum(seqlens))
    q, k, v = (
        torch.randn(total, h, d, generator=g, dtype=torch.float32).to(dtype).to(device) for _ in range(3)
    )
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(seqlens), 0)
    return q, k, v, cu, torch.tensor(prefix_lens, dtype=torch.int32)
