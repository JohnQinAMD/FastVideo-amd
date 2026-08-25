"""Grouped/multi-query attention coverage for the CK VSA forward kernel.

The CK kernel picks a KV head as ``i_nhead / (nhead_q / nhead_k)``, so K/V may
carry fewer heads than Q. The block LUT stays indexed by *query* head in every
case. These tests pin that contract down, since a stride mistake here produces
plausible-looking but wrong output rather than a failure.
"""
import pytest
import torch

BLOCK_KV = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


def _ck_fwd():
    try:
        from fastvideo_kernel.ck_sparse_attn import ck_block_sparse_attn_fwd
        from fastvideo_kernel.ck_sparse_attn import _load_ck_extension

        _load_ck_extension()
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"CK VSA extension unavailable: {e}")
    return ck_block_sparse_attn_fwd


def build_lut(B, Hq, Sq, Sk, block_m, density, device, seed=0):
    """Top-K block selection per query head, laid out the way the kernel reads it.

    Row width is fixed at ceil(Sk/64) because the kernel strides LUT rows by
    that amount; selected blocks are packed at the front in ascending order and
    the tail past ``counts`` is -1.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    q_blks = Sq // block_m
    kv_blks = Sk // BLOCK_KV
    keep = max(1, int(round(kv_blks * density)))

    scores = torch.rand(B, Hq, q_blks, kv_blks, generator=g)
    block_map = torch.zeros(B, Hq, q_blks, kv_blks, dtype=torch.bool)
    block_map.scatter_(-1, scores.topk(keep, dim=-1).indices, True)

    counts = block_map.sum(-1).to(torch.int32)
    order = torch.argsort(block_map.to(torch.int8), dim=-1, descending=True,
                          stable=True)
    ar = torch.arange(kv_blks, dtype=torch.int32)
    valid = ar.view(1, 1, 1, -1) < counts.unsqueeze(-1)
    idx = torch.where(valid, order.to(torch.int32),
                      torch.full_like(order, -1, dtype=torch.int32))
    return idx.contiguous().to(device), counts.contiguous().to(device), kv_blks


def reference(q, k, v, idx, num, block_m, vbs=None):
    """Dense fp32 attention restricted to the selected blocks.

    When ``vbs`` is given, only the first ``vbs[blk]`` tokens of each KV block
    are real; the rest are padding and must not enter the softmax.
    """
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    group = Hq // Hkv

    qf, kf, vf = q.float(), k.float(), v.float()
    scale = D ** -0.5
    out = torch.empty(B, Hq, Sq, D, dtype=torch.float32, device=q.device)

    for b in range(B):
        for h in range(Hq):
            kh = h // group
            mask = torch.zeros(Sq, Sk, dtype=torch.bool, device=q.device)
            for m in range(Sq // block_m):
                for blk in idx[b, h, m, : num[b, h, m]].tolist():
                    real = BLOCK_KV if vbs is None else int(vbs[blk])
                    mask[m * block_m : (m + 1) * block_m,
                         blk * BLOCK_KV : blk * BLOCK_KV + real] = True
            s = (qf[b, h] @ kf[b, kh].transpose(-1, -2)) * scale
            out[b, h] = s.masked_fill(~mask, float("-inf")).softmax(-1) @ vf[b, kh]
    return out


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 2), (8, 1), (12, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_m", [64, 128])
def test_gqa_matches_dense_reference(Hq, Hkv, dtype, block_m):
    ck_fwd = _ck_fwd()
    B, Sq, Sk, D = 1, 512, 512, 128
    torch.manual_seed(1234)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.25, "cuda")
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o, _ = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
    ref = reference(q, k, v, idx, num, block_m)

    assert cos_sim(o, ref) > 0.999
    assert (o.float() - ref).norm() / ref.norm() < 2e-2


@pytest.mark.parametrize("block_m", [64, 128])
def test_gqa_with_variable_block_sizes(block_m):
    """GQA on top of ragged KV blocks, which also runs the correction kernel."""
    ck_fwd = _ck_fwd()
    B, Hq, Hkv, Sq, Sk, D = 1, 8, 2, 512, 512, 128
    torch.manual_seed(99)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.25, "cuda", seed=3)
    g = torch.Generator(device="cpu").manual_seed(11)
    vbs = torch.randint(32, BLOCK_KV + 1, (kv_blks,), generator=g,
                        dtype=torch.int32)
    for blk in range(kv_blks):
        real = int(vbs[blk])
        k[:, :, blk * BLOCK_KV + real : (blk + 1) * BLOCK_KV] = 0
        v[:, :, blk * BLOCK_KV + real : (blk + 1) * BLOCK_KV] = 0

    o, _ = ck_fwd(q, k, v, idx, num, vbs.cuda(), block_m,
                  skip_vbs_correction=False)
    ref = reference(q, k, v, idx, num, block_m, vbs=vbs)

    assert cos_sim(o, ref) > 0.999
    assert (o.float() - ref).norm() / ref.norm() < 5e-2


def test_gqa_equals_replicated_kv():
    """GQA must reproduce the MHA result on explicitly replicated K/V.

    A per-head reference can hide a KV stride bug if it makes the same mistake;
    comparing against the MHA path cannot.
    """
    ck_fwd = _ck_fwd()
    B, Hq, Hkv, Sq, Sk, D, block_m = 1, 8, 2, 512, 512, 128, 64
    torch.manual_seed(7)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.3, "cuda", seed=5)
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o_gqa, _ = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
    o_mha, _ = ck_fwd(q,
                      k.repeat_interleave(Hq // Hkv, dim=1).contiguous(),
                      v.repeat_interleave(Hq // Hkv, dim=1).contiguous(),
                      idx, num, vbs, block_m, skip_vbs_correction=True)

    assert torch.equal(o_gqa, o_mha)


def test_indivisible_head_count_is_rejected():
    ck_fwd = _ck_fwd()
    B, Hq, Hkv, Sq, Sk, D, block_m = 1, 6, 4, 512, 512, 128, 64
    torch.manual_seed(0)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.5, "cuda")
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="divisible"):
        ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
