#!/usr/bin/env python3
"""Quick test + benchmark of CK VSA fwd vs Triton fwd."""
import sys, os, math, time
import torch

# Add CK .so to path
CK_LIB = os.path.join(os.path.dirname(__file__), "..", "csrc", "attention", "ck_sparse", "build")
sys.path.insert(0, CK_LIB)
# Fix LD_LIBRARY_PATH for torch if needed
from torch.utils.cpp_extension import library_paths
for p in library_paths():
    os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

import ck_vsa_ops
from fastvideo_kernel.block_sparse_attn import _map_to_index
from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward

BLOCK = 64

def make_inputs(B=1, H=12, Sq=4096, D=128, topk=6):
    Sk = Sq
    q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device="cuda")
    nq, nkv = Sq // BLOCK, Sk // BLOCK
    scores = torch.rand(B, H, nq, nkv, device="cuda")
    idx = torch.topk(scores, min(topk, nkv), dim=-1).indices
    bm = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
    bm.scatter_(-1, idx, True)
    q2k_idx, q2k_num = _map_to_index(bm)
    vbs = torch.full((nkv,), BLOCK, dtype=torch.int32, device="cuda")
    return q, k, v, q2k_idx, q2k_num, vbs

def test_correctness():
    print("=== Correctness test (small) ===")
    torch.manual_seed(42)
    q, k, v, q2k_idx, q2k_num, vbs = make_inputs(B=1, H=2, Sq=256, D=128, topk=2)

    # Triton reference
    o_tri, M_tri = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
    torch.cuda.synchronize()

    # CK
    o_ck, lse_ck = ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, 64)
    torch.cuda.synchronize()

    # Compare outputs
    diff = (o_tri.float() - o_ck.float()).abs()
    rel_err = diff.max() / o_tri.float().abs().mean()
    print(f"  Triton out norm: {o_tri.float().norm():.4f}")
    print(f"  CK out norm:     {o_ck.float().norm():.4f}")
    print(f"  Max abs diff:    {diff.max():.6f}")
    print(f"  Rel max diff:    {rel_err:.6f}")
    if rel_err < 0.05:
        print("  PASS")
    else:
        print("  FAIL (rel_err too large)")
    return rel_err < 0.05

def benchmark():
    print("\n=== Benchmark: CK vs Triton ===")
    from triton.testing import do_bench

    configs = [
        (1, 12, 4096, 128, 6),
        (1, 12, 16384, 128, 25),
        (1, 12, 49152, 128, 76),
    ]
    print(f"{'Sq':>8} {'topk':>5} {'Triton ms':>10} {'CK ms':>10} {'Speedup':>8}")
    print("-" * 50)

    for B, H, Sq, D, topk in configs:
        q, k, v, q2k_idx, q2k_num, vbs = make_inputs(B, H, Sq, D, topk)

        ms_tri = do_bench(lambda: triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs),
                          warmup=5, rep=20)

        ms_ck = do_bench(lambda: ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, 64),
                         warmup=5, rep=20)

        speedup = ms_tri / ms_ck
        print(f"{Sq:8d} {topk:5d} {ms_tri:10.3f} {ms_ck:10.3f} {speedup:8.2f}x")

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ok = test_correctness()
    if ok:
        benchmark()
    else:
        print("Skipping benchmark due to correctness failure.")
