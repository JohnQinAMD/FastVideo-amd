"""Regression coverage for KV-block padding in the CK VSA forward kernel.

VSA pads every KV block out to 64 tokens with zero K/V. A zero K row scores
exactly 0, so padding is only harmless while some real token in the row scores
above 0 - then the row max is set by real data and the padded slots contribute
exp(0 - m) < 1 each. The moment *every* real score in a row is negative the row
max is 0, each padded slot contributes exactly 1, and the denominator is
dominated by however many padded slots the selected blocks happen to carry.

That case is what these tests build. Random Gaussian q/k never reach it - with
D=128 and unit-variance inputs some selected key almost surely scores positive -
which is why the suite in test_ck_vsa_gqa.py passed against a kernel that scored
padding and rescaled afterwards. Rescaling cannot recover it in fp32: the real
mass is a part in thousands of the denominator here, so backing the padding out
means subtracting two numbers that agree to more digits than fp32 carries.

The kernel now masks padded columns before the softmax instead. Each test
therefore also asserts the *unmasked* instance gets the same input badly wrong,
so a regression that stops masking cannot pass by accident.
"""
import math

import pytest
import torch

from .test_ck_vsa_gqa import BLOCK_KV, _ck_fwd, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


def build_padding_dominant_case(B, H, Sq, Sk, D, block_m, valid, dtype):
    """Inputs whose real scores are all negative, so padding owns the softmax.

    Every query attends every KV block, and only the first ``valid`` tokens of
    each block are real. Q points opposite to K along one axis, which puts every
    real score at a definite negative value while padding sits at exactly 0.
    """
    dev = "cuda"
    kv_blks = Sk // BLOCK_KV

    q = torch.zeros(B, H, Sq, D, device=dev, dtype=dtype)
    k = torch.zeros(B, H, Sk, D, device=dev, dtype=dtype)
    # Spread the real scores over a range instead of making them all equal, so
    # the softmax has actual structure to get wrong.
    q[..., 0] = -8.0
    for blk in range(kv_blks):
        base = blk * BLOCK_KV
        for j in range(valid):
            k[:, :, base + j, 0] = 4.0 + (j % 4)

    # Distinct V per real row, so a diluted denominator shows up as a scale
    # error rather than being hidden by averaging identical values.
    g = torch.Generator(device="cpu").manual_seed(7)
    v = torch.zeros(B, H, Sk, D, dtype=torch.float32)
    for blk in range(kv_blks):
        base = blk * BLOCK_KV
        v[:, :, base : base + valid] = torch.randn(
            B, H, valid, D, generator=g
        )
    v = v.to(device=dev, dtype=dtype)

    q_blks = Sq // block_m
    idx = torch.arange(kv_blks, dtype=torch.int32, device=dev)
    idx = idx.view(1, 1, 1, -1).expand(B, H, q_blks, kv_blks).contiguous()
    num = torch.full((B, H, q_blks), kv_blks, dtype=torch.int32, device=dev)
    vbs = torch.full((kv_blks,), valid, dtype=torch.int32)
    return q, k, v, idx, num, vbs


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_padding_does_not_reach_the_softmax(D, block_m, dtype):
    ck_fwd = _ck_fwd()
    B, H, Sq, Sk, valid = 2, 4, block_m * 2, 256, 8

    q, k, v, idx, num, vbs = build_padding_dominant_case(
        B, H, Sq, Sk, D, block_m, valid, dtype
    )
    ref, ref_lse = reference(q, k, v, idx, num, block_m, vbs=vbs)

    o, lse = ck_fwd(q, k, v, idx, num, vbs.cuda(), block_m,
                    uniform_block_sizes=False)

    of = o.float()
    assert torch.isfinite(of).all(), "output contains NaN or Inf"
    scale = ref.abs().amax().clamp_min(1e-6)
    worst = ((of - ref).abs().amax() / scale).item()
    assert worst < 2e-2, f"worst element off by {worst:.4f} of scale"
    assert (lse - ref_lse).abs().amax().item() < 1e-2

    # Confirm the case is actually hostile: 56 of every 64 slots are padding, so
    # an instance that scores them lands nowhere near the reference. Without
    # this the test above could pass on an input that never stressed anything.
    o_unmasked, lse_unmasked = ck_fwd(q, k, v, idx, num, vbs.cuda(), block_m,
                                      uniform_block_sizes=True)
    diluted = ((o_unmasked.float() - ref).abs().amax() / scale).item()
    assert diluted > 0.5, (
        f"unmasked instance was only off by {diluted:.4f} of scale, so this "
        "input no longer reproduces the padding-dominated denominator"
    )
    pad_per_row = (Sk // BLOCK_KV) * (BLOCK_KV - valid)
    assert lse_unmasked.amax().item() > math.log2(pad_per_row) - 1.0


def test_single_valid_token_per_block():
    """The extreme of the same failure: 63 of every 64 slots are padding."""
    ck_fwd = _ck_fwd()
    B, H, Sq, Sk, D, block_m, valid = 1, 2, 64, 256, 128, 64, 1

    q, k, v, idx, num, vbs = build_padding_dominant_case(
        B, H, Sq, Sk, D, block_m, valid, torch.bfloat16
    )
    ref, ref_lse = reference(q, k, v, idx, num, block_m, vbs=vbs)

    o, lse = ck_fwd(q, k, v, idx, num, vbs.cuda(), block_m,
                    uniform_block_sizes=False)

    of = o.float()
    assert torch.isfinite(of).all()
    scale = ref.abs().amax().clamp_min(1e-6)
    assert ((of - ref).abs().amax() / scale).item() < 2e-2
    assert (lse - ref_lse).abs().amax().item() < 1e-2


def test_block_size_table_must_cover_the_padded_sequence():
    """The mask indexes the table by absolute block, so a short table is fatal."""
    ck_fwd = _ck_fwd()
    q, k, v, idx, num, vbs = build_padding_dominant_case(
        1, 2, 64, 256, 128, 64, 8, torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="one entry per KV block"):
        ck_fwd(q, k, v, idx, num, vbs[:-1].cuda(), 64,
               uniform_block_sizes=False)
