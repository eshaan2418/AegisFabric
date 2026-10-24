import math
import os

import pytest
import torch

from kernels import prefix_attn as K

CASES = [  # (seqlens, prefix_lens, H, D)
    ([5], [0], 1, 16),
    ([17, 1, 40], [4, 0, 16], 2, 32),
    ([64, 33, 100, 7], [16, 16, 16, 7], 4, 32),
    ([130, 64], [16, 64], 2, 64),
]
# Tolerances: max |err| vs fp64 reference. fp32 accumulation over <=130 keys.
TOL = {torch.float32: 2e-5, torch.float16: 2e-3}


def test_mask_semantics():
    q, k, v, cu, pl = K.make_inputs([6], [3], 1, 16, dtype=torch.float64)
    # attention output of a prefix query must equal attention over exactly the prefix (bidirectional)
    out = K.reference_packed(q, k, v, cu, pl)
    s = (q[1, 0] @ k[:3, 0].T) / math.sqrt(16)
    exp = torch.softmax(s, -1) @ v[:3, 0]
    assert torch.allclose(out[1, 0], exp)
    # text query 4 sees keys 0..4
    s = (q[4, 0] @ k[:5, 0].T) / math.sqrt(16)
    assert torch.allclose(out[4, 0], torch.softmax(s, -1) @ v[:5, 0])


@pytest.mark.parametrize("case", CASES)
def test_padded_unfused_matches_reference(case):
    seqlens, pls, h, d = case
    q, k, v, cu, pl = K.make_inputs(seqlens, pls, h, d, dtype=torch.float64)
    ref = K.reference_packed(q, k, v, cu, pl)
    L = max(seqlens)
    mask = K.padded_mask(torch.tensor(seqlens), pl, L)
    out = K.padded_unfused(*(K.pack_to_padded(t, cu, L) for t in (q, k, v)), mask, 1 / math.sqrt(d))
    assert torch.allclose(K.padded_to_pack(out, cu), ref, atol=1e-10)


triton_ok = K.HAVE_TRITON and os.environ.get("TRITON_INTERPRET") == "1"


@pytest.mark.skipif(not triton_ok, reason="needs triton with TRITON_INTERPRET=1 (Linux lab container)")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("blocks", [(16, 16), (32, 16)])
def test_triton_matches_reference(case, dtype, blocks):
    seqlens, pls, h, d = case
    q, k, v, cu, pl = K.make_inputs(seqlens, pls, h, d, dtype=dtype, seed=len(seqlens))
    ref = K.reference_packed(q.double(), k.double(), v.double(), cu, pl)
    out = K.triton_prefix_attn(q, k, v, cu, pl, max(seqlens), block_m=blocks[0], block_n=blocks[1])
    err = (out.double() - ref).abs().max().item()
    assert err <= TOL[dtype], err


mps_ok = torch.backends.mps.is_available()


@pytest.mark.skipif(not mps_ok, reason="needs Apple MPS")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_metal_matches_reference(case, dtype):
    from kernels.metal_prefix_attn import metal_prefix_attn

    seqlens, pls, h, d = case
    q, k, v, cu, pl = K.make_inputs(seqlens, pls, h, d, dtype=dtype, seed=len(seqlens))
    ref = K.reference_packed(q.double(), k.double(), v.double(), cu, pl)
    out = metal_prefix_attn(q.to("mps"), k.to("mps"), v.to("mps"), cu, pl).cpu()
    err = (out.double() - ref).abs().max().item()
    assert err <= TOL[dtype], err
