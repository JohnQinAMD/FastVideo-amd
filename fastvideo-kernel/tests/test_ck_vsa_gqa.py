"""Grouped/multi-query attention coverage for the CK VSA forward kernel.

The CK kernel picks a KV head as ``i_nhead / (nhead_q / nhead_k)``, so K/V may
carry fewer heads than Q. The block LUT stays indexed by *query* head in every
case. These tests pin that contract down, since a stride mistake here produces
plausible-looking but wrong output rather than a failure.

Correctness is judged three ways, because a cosine similarity over millions of
elements hides localized errors:

* against an fp32 dense reference, on worst-element error rather than only an
  aggregate, and on LSE, which the kernel computes separately and which matches
  to ~1e-6 rather than the ~2e-3 of a bf16 output;
* against the MHA path on explicitly replicated K/V, which must be bit-identical;
* against constructed inputs whose output names the (batch, KV head) it was read
  from, so the addressing is checked categorically instead of statistically.

Every batch dimension here is deliberately > 1 in at least one case per code
path: the K/V batch stride is derived from the KV head count, and a B=1 test
cannot observe it.
"""
import math

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


def _block_mask(idx, num, b, h, Sq, Sk, block_m, vbs, device):
    mask = torch.zeros(Sq, Sk, dtype=torch.bool, device=device)
    for m in range(Sq // block_m):
        for blk in idx[b, h, m, : num[b, h, m]].tolist():
            real = BLOCK_KV if vbs is None else int(vbs[blk])
            mask[m * block_m : (m + 1) * block_m,
                 blk * BLOCK_KV : blk * BLOCK_KV + real] = True
    return mask


def reference(q, k, v, idx, num, block_m, vbs=None):
    """Dense fp32 attention restricted to the selected blocks, plus its LSE.

    When ``vbs`` is given, only the first ``vbs[blk]`` tokens of each KV block
    are real; the rest are padding and must not enter the softmax. LSE is
    returned in the log2 domain to match the kernel's exp2 convention.
    """
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    group = Hq // Hkv

    qf, kf, vf = q.float(), k.float(), v.float()
    scale = D ** -0.5
    out = torch.empty(B, Hq, Sq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(B, Hq, Sq, dtype=torch.float32, device=q.device)

    for b in range(B):
        for h in range(Hq):
            kh = h // group
            mask = _block_mask(idx, num, b, h, Sq, Sk, block_m, vbs, q.device)
            s = ((qf[b, h] @ kf[b, kh].transpose(-1, -2)) * scale)
            s = s.masked_fill(~mask, float("-inf"))
            out[b, h] = s.softmax(-1) @ vf[b, kh]
            lse[b, h] = torch.logsumexp(s, -1) / math.log(2.0)
    return out, lse


def assert_close(o, lse, ref, ref_lse, tol=2e-2, lse_tol=1e-3):
    """Judge on worst element and on LSE, not on an aggregate over all elements."""
    of = o.float()
    assert torch.isfinite(of).all(), "output contains NaN or Inf"

    # Attention output passes through zero, so normalize the error by the
    # tensor's scale instead of taking an elementwise ratio.
    scale = ref.abs().amax().clamp_min(1e-6)
    worst = (of - ref).abs().amax() / scale
    assert worst < tol, f"worst element off by {worst:.4f} of scale (tol {tol})"

    # Per head, so an error confined to one head is not diluted by the others.
    for b in range(ref.shape[0]):
        for h in range(ref.shape[1]):
            a, c = of[b, h].flatten(), ref[b, h].flatten()
            cs = (a @ c / (a.norm() * c.norm())).item()
            assert cs > 0.9999, f"head (b={b}, h={h}) cos_sim {cs:.6f}"

    # The kernel accumulates LSE separately in fp32, so this is a tighter and
    # largely independent check on the softmax denominator.
    assert (lse - ref_lse).abs().amax() < lse_tol


# (B, Hq, Hkv) — B > 1 is required to exercise the K/V batch stride.
SHAPES = [(1, 8, 8), (1, 8, 2), (1, 8, 1), (2, 8, 2), (3, 12, 3), (2, 8, 1)]


@pytest.mark.parametrize("B,Hq,Hkv", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_m", [64, 128])
def test_gqa_matches_dense_reference(B, Hq, Hkv, dtype, block_m):
    ck_fwd = _ck_fwd()
    Sq = Sk = 512
    D = 128
    torch.manual_seed(1234)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dtype)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.25, "cuda")
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o, lse = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
    ref, ref_lse = reference(q, k, v, idx, num, block_m)
    assert_close(o, lse, ref, ref_lse)


@pytest.mark.parametrize("B,Hq,Hkv,Sq,Sk", [
    (2, 8, 2, 512, 1024),
    (2, 8, 2, 1024, 512),
    (1, 8, 1, 256, 1024),
])
def test_gqa_with_asymmetric_q_kv_lengths(B, Hq, Hkv, Sq, Sk):
    """Sq != Sk, so that a K/V head stride taken from Sq instead of Sk shows up."""
    ck_fwd = _ck_fwd()
    D, block_m = 128, 64
    torch.manual_seed(21)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.25, "cuda", seed=4)
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o, lse = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
    ref, ref_lse = reference(q, k, v, idx, num, block_m)
    assert_close(o, lse, ref, ref_lse)


@pytest.mark.parametrize("block_m", [64, 128])
def test_gqa_with_variable_block_sizes(block_m):
    """GQA on top of ragged KV blocks, which also runs the correction kernel."""
    ck_fwd = _ck_fwd()
    B, Hq, Hkv, Sq, Sk, D = 2, 8, 2, 512, 512, 128
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
    ref, ref_lse = reference(q, k, v, idx, num, block_m, vbs=vbs)

    # LSE is checked by the uniform-block tests; here the kernel deliberately
    # rewrites it to remove the padded positions, so only the output is compared.
    of = o.float()
    scale = ref.abs().amax().clamp_min(1e-6)
    assert torch.isfinite(of).all()
    assert (of - ref).abs().amax() / scale < 5e-2


@pytest.mark.parametrize("B,Hq,Hkv", [(1, 8, 2), (3, 8, 2), (2, 8, 1)])
def test_gqa_equals_replicated_kv(B, Hq, Hkv):
    """GQA must reproduce the MHA path run on explicitly replicated K/V.

    A per-head reference can hide a KV addressing bug by repeating the same
    mistake; an exact comparison against the already-trusted MHA path cannot.
    """
    ck_fwd = _ck_fwd()
    Sq = Sk = 512
    D, block_m = 128, 64
    torch.manual_seed(7)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.3, "cuda", seed=5)
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o_gqa, lse_gqa = ck_fwd(q, k, v, idx, num, vbs, block_m,
                            skip_vbs_correction=True)
    o_mha, lse_mha = ck_fwd(q,
                            k.repeat_interleave(Hq // Hkv, dim=1).contiguous(),
                            v.repeat_interleave(Hq // Hkv, dim=1).contiguous(),
                            idx, num, vbs, block_m, skip_vbs_correction=True)

    assert torch.equal(o_gqa, o_mha)
    assert torch.equal(lse_gqa, lse_mha)


@pytest.mark.parametrize("B,Hq,Hkv", [(3, 8, 2), (2, 12, 3), (3, 8, 1)])
def test_output_identifies_the_kv_head_it_read(B, Hq, Hkv):
    """Give every (batch, KV head) a constant V and check the value that comes back.

    Softmax weights sum to one, so with V constant across a head the output must
    equal that constant exactly, whatever the mask or the scores. The result is
    therefore a label naming which (batch, KV head) the kernel addressed, which
    turns a statistical comparison into a categorical one: an off-by-one head or
    a batch stride derived from the wrong head count returns a different integer.
    """
    ck_fwd = _ck_fwd()
    Sq = Sk = 512
    D, block_m = 128, 64
    group = Hq // Hkv
    torch.manual_seed(3)

    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    labels = (torch.arange(B * Hkv, device="cuda", dtype=torch.float32) + 1)
    v = labels.view(B, Hkv, 1, 1).expand(B, Hkv, Sk, D).to(torch.bfloat16).contiguous()

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.25, "cuda", seed=8)
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o, _ = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)

    expected = torch.stack([
        torch.stack([labels[b * Hkv + h // group].expand(Sq, D) for h in range(Hq)])
        for b in range(B)
    ])
    assert torch.allclose(o.float(), expected, atol=1e-2), (
        f"read the wrong (batch, KV head): got {o.float()[..., 0].unique().tolist()[:8]}, "
        f"expected {expected[..., 0].unique().tolist()[:8]}"
    )


@pytest.mark.parametrize("B,Hq,Hkv", [(3, 8, 2), (2, 8, 1)])
def test_output_identifies_the_k_head_it_read(B, Hq, Hkv):
    """Same idea for K: one dominant key per (batch, KV head), at a distinct index.

    V encodes the key's position, so the output reports which K row won the
    softmax and therefore which K head was addressed.
    """
    ck_fwd = _ck_fwd()
    Sq = Sk = 512
    D, block_m = 128, 64
    group = Hq // Hkv

    q = torch.ones(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    spikes = torch.zeros(B, Hkv, dtype=torch.long)
    for b in range(B):
        for j in range(Hkv):
            spikes[b, j] = (b * Hkv + j) * 37 + 5  # distinct, and exact in bf16
            k[b, j, spikes[b, j], :] = 4.0

    pos = torch.arange(Sk, device="cuda", dtype=torch.float32)
    v = torch.zeros(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v[:, :, :, 0] = pos.view(1, 1, Sk).to(torch.bfloat16)

    # Every block selected, so the spike is always visible to every query block.
    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 1.0, "cuda", seed=2)
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    o, _ = ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)

    got = o.float()[:, :, :, 0]
    for b in range(B):
        for h in range(Hq):
            want = float(spikes[b, h // group])
            assert (got[b, h] - want).abs().max() < 1.0, (
                f"b={b} h={h} attended to key ~{got[b, h].mean():.1f}, "
                f"expected {want:.0f}"
            )


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


def test_mismatched_kv_shapes_are_rejected():
    ck_fwd = _ck_fwd()
    B, Hq, Hkv, Sq, Sk, D, block_m = 1, 8, 2, 512, 512, 128, 64
    torch.manual_seed(0)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv * 2, Sk, D, device="cuda", dtype=torch.bfloat16)

    idx, num, kv_blks = build_lut(B, Hq, Sq, Sk, block_m, 0.5, "cuda")
    vbs = torch.full((kv_blks,), BLOCK_KV, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="same shape"):
        ck_fwd(q, k, v, idx, num, vbs, block_m, skip_vbs_correction=True)
