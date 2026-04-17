#!/usr/bin/env python3
"""
Profile VSA block-sparse attention kernels with rocprof PMC counters.

Isolates the Triton kernel dispatch (no wrapper overhead) for accurate PMC.
Usage:
  # Forward only:
  rocprofv3 --pmc SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VMEM SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY -- python3 profile_vsa.py --mode fwd
  # Backward only:
  rocprofv3 --pmc SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VMEM SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY -- python3 profile_vsa.py --mode bwd
  # Both:
  rocprofv3 --pmc SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VMEM SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY -- python3 profile_vsa.py --mode both
"""
from __future__ import annotations
import argparse, math, os, random
import numpy as np
import torch

os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

from fastvideo_kernel.block_sparse_attn import _map_to_index
from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (
    triton_block_sparse_attn_forward,
    triton_block_sparse_attn_backward,
)

BLOCK = 64

def make_inputs(bs, h, q_len, kv_len, d, topk, dtype=torch.bfloat16):
    q = torch.randn(bs, h, q_len, d, dtype=dtype, device="cuda")
    k = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")
    v = torch.randn(bs, h, kv_len, d, dtype=dtype, device="cuda")

    num_q = q_len // BLOCK
    num_kv = kv_len // BLOCK
    topk = min(max(1, topk), num_kv)

    scores = torch.rand(bs, h, num_q, num_kv, device="cuda")
    idx = torch.topk(scores, topk, dim=-1).indices
    block_map = torch.zeros(bs, h, num_q, num_kv, dtype=torch.bool, device="cuda")
    block_map.scatter_(-1, idx, True)

    q2k_idx, q2k_num = _map_to_index(block_map)
    k2q_idx, k2q_num = _map_to_index(block_map.transpose(-1, -2).contiguous())
    vbs = torch.full((num_kv,), BLOCK, dtype=torch.int32, device="cuda")
    return q, k, v, block_map, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs


def profile_fwd(q, k, v, q2k_idx, q2k_num, vbs, warmup=5, rep=3):
    # warmup
    for _ in range(warmup):
        triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
    torch.cuda.synchronize()
    # measured iterations (rocprof captures these)
    for _ in range(rep):
        o, M = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
    torch.cuda.synchronize()
    return o, M


def profile_bwd(do, q, k, v, o, M, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs, warmup=5, rep=3):
    for _ in range(warmup):
        triton_block_sparse_attn_backward(do, q, k, v, o, M, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs)
    torch.cuda.synchronize()
    for _ in range(rep):
        triton_block_sparse_attn_backward(do, q, k, v, o, M, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs)
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fwd", "bwd", "both"], default="both")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--q_len", type=int, default=49152)
    p.add_argument("--kv_len", type=int, default=49152)
    p.add_argument("--topk", type=int, default=76)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=3)
    args = p.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    q, k, v, bm, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs = make_inputs(
        args.bs, args.heads, args.q_len, args.kv_len, args.head_dim, args.topk
    )
    print(f"Config: bs={args.bs} h={args.heads} d={args.head_dim} "
          f"q={args.q_len} kv={args.kv_len} topk={args.topk}")

    if args.mode in ("fwd", "both"):
        print("--- Forward ---")
        o, M = profile_fwd(q, k, v, q2k_idx, q2k_num, vbs, args.warmup, args.rep)
        print("  fwd done")

    if args.mode in ("bwd", "both"):
        if args.mode == "bwd":
            o, M = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
            torch.cuda.synchronize()
        do = torch.randn_like(o)
        print("--- Backward ---")
        profile_bwd(do, q, k, v, o, M, q2k_idx, q2k_num, k2q_idx, k2q_num, vbs, args.warmup, args.rep)
        print("  bwd done")

    print("Profiling complete.")


if __name__ == "__main__":
    main()
