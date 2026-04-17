#!/usr/bin/env python3
"""
Strict correctness + performance test suite for CK VSA block-sparse attention.

Tests:
  1. Numerical correctness vs PyTorch reference (fp32 masked attention)
  2. Numerical correctness vs Triton implementation
  3. Edge cases: single block, full density, single head, variable topk
  4. Multiple sequence lengths, head counts, head dims
  5. fp16 and bf16 dtypes
  6. block_m=64 and block_m=128 tile configs
  7. Asymmetric Q/KV lengths
  8. Performance benchmark sweep

Usage:
  python3 strict_test_ck_fwd.py              # full suite
  python3 strict_test_ck_fwd.py --quick      # correctness only (fast)
  python3 strict_test_ck_fwd.py --bench-only # benchmark only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import torch
import numpy as np

# Setup paths
CK_LIB = os.path.join(os.path.dirname(__file__), "..", "csrc", "attention", "ck_sparse", "build")
sys.path.insert(0, CK_LIB)
os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

from torch.utils.cpp_extension import library_paths
for p in library_paths():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if p not in ld:
        os.environ["LD_LIBRARY_PATH"] = p + ":" + ld

import ck_vsa_ops
from fastvideo_kernel.block_sparse_attn import _map_to_index
from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward

BLOCK = 64


# ─────────────────────────── Helpers ───────────────────────────────────────

def pytorch_reference(q, k, v, block_map, variable_block_sizes):
    """
    Dense PyTorch reference: fp32 masked attention.
    block_map: [B, H, num_q_blocks, num_kv_blocks] bool
    variable_block_sizes: [num_kv_blocks] int32 — tokens per KV block
    """
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    nq = Sq // BLOCK
    nkv = Sk // BLOCK

    # Build full token-level mask from block map (per-head)
    full_mask = torch.zeros(B, H, Sq, Sk, dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qi in range(nq):
                for ki in range(nkv):
                    if block_map[b, h, qi, ki]:
                        bs = variable_block_sizes[ki].item()
                        full_mask[b, h, qi*BLOCK:(qi+1)*BLOCK, ki*BLOCK:ki*BLOCK+bs] = True

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scale = 1.0 / math.sqrt(D)

    qk = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    qk = qk.masked_fill(~full_mask, float('-inf'))
    attn = torch.nn.functional.softmax(qk, dim=-1)
    attn = attn.masked_fill(torch.isnan(attn), 0.0)
    out = torch.matmul(attn, v_f)
    return out.to(q.dtype)


def make_inputs(B, H, Sq, D, Sk, topk, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, Sk, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, Sk, D, dtype=dtype, device="cuda")

    nq = Sq // BLOCK
    nkv = Sk // BLOCK
    topk = min(max(1, topk), nkv)

    scores = torch.rand(B, H, nq, nkv, device="cuda")
    idx = torch.topk(scores, topk, dim=-1).indices
    block_map = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
    block_map.scatter_(-1, idx, True)

    q2k_idx, q2k_num = _map_to_index(block_map)
    vbs = torch.full((nkv,), BLOCK, dtype=torch.int32, device="cuda")

    return q, k, v, block_map, q2k_idx, q2k_num, vbs


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_diff: float
    rel_max_diff: float
    avg_abs_diff: float
    config: str
    note: str = ""


@dataclass
class BenchResult:
    config: str
    Sq: int
    Sk: int
    topk: int
    heads: int
    head_dim: int
    dtype: str
    block_m: int
    triton_ms: float
    ck_ms: float
    speedup: float
    ck_tflops: float


# ─────────────────────────── Correctness Tests ─────────────────────────────

def test_vs_pytorch(B, H, Sq, D, Sk, topk, dtype, block_m, seed=42) -> TestResult:
    """Compare CK output against PyTorch fp32 reference."""
    name = f"pytorch_ref B={B} H={H} Sq={Sq} Sk={Sk} D={D} topk={topk} {str(dtype).split('.')[-1]} bm={block_m}"
    q, k, v, bm, q2k_idx, q2k_num, vbs = make_inputs(B, H, Sq, D, Sk, topk, dtype, seed)

    ref = pytorch_reference(q, k, v, bm, vbs)
    o_ck, lse_ck = ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, block_m)
    torch.cuda.synchronize()

    diff = (ref.float() - o_ck.float()).abs()
    ref_abs_mean = ref.float().abs().mean().item()
    max_abs = diff.max().item()
    rel_max = max_abs / max(ref_abs_mean, 1e-8)
    avg_abs = diff.mean().item()

    # bf16 vs fp32 ref tolerance: 5% for bf16, 2% for fp16
    # (bf16 has 8-bit mantissa; flash-attention-style online softmax adds numerical noise)
    tol = 0.05 if dtype == torch.bfloat16 else 0.02
    passed = rel_max < tol

    return TestResult(name=name, passed=passed, max_abs_diff=max_abs,
                      rel_max_diff=rel_max, avg_abs_diff=avg_abs,
                      config=f"Sq={Sq},Sk={Sk},topk={topk},D={D},H={H},{str(dtype).split('.')[-1]},bm={block_m}",
                      note="" if passed else f"FAIL: rel_max={rel_max:.6f} > tol={tol}")


def test_vs_triton(B, H, Sq, D, Sk, topk, dtype, block_m, seed=42) -> TestResult:
    """Compare CK output against Triton implementation."""
    name = f"triton_ref B={B} H={H} Sq={Sq} Sk={Sk} D={D} topk={topk} {str(dtype).split('.')[-1]} bm={block_m}"
    q, k, v, bm, q2k_idx, q2k_num, vbs = make_inputs(B, H, Sq, D, Sk, topk, dtype, seed)

    o_tri, M_tri = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
    o_ck, lse_ck = ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, block_m)
    torch.cuda.synchronize()

    diff = (o_tri.float() - o_ck.float()).abs()
    ref_abs_mean = o_tri.float().abs().mean().item()
    max_abs = diff.max().item()
    rel_max = max_abs / max(ref_abs_mean, 1e-8)
    avg_abs = diff.mean().item()

    # Two independent bf16 implementations (CK vs Triton): different online-softmax
    # accumulation order and rounding → expect up to ~5% relative max error.
    tol = 0.05
    passed = rel_max < tol

    return TestResult(name=name, passed=passed, max_abs_diff=max_abs,
                      rel_max_diff=rel_max, avg_abs_diff=avg_abs,
                      config=f"Sq={Sq},Sk={Sk},topk={topk},D={D},H={H},{str(dtype).split('.')[-1]},bm={block_m}",
                      note="" if passed else f"FAIL: rel_max={rel_max:.6f} > tol={tol}")


def run_correctness_tests() -> List[TestResult]:
    results = []

    print("=" * 80)
    print("STRICT CORRECTNESS TESTS — CK VSA block-sparse attention forward")
    print("=" * 80)

    # ---- Test 1: Small configs vs PyTorch reference ----
    print("\n[1/6] Small configs vs PyTorch fp32 reference")
    small_configs = [
        # (B, H, Sq, D, Sk, topk, dtype, block_m)
        (1, 1, 128, 128, 128, 1, torch.bfloat16, 64),    # minimal: 1 head, 1 block topk
        (1, 1, 128, 128, 128, 2, torch.bfloat16, 64),    # 2 blocks, full density
        (1, 2, 256, 128, 256, 2, torch.bfloat16, 64),    # 2 heads
        (1, 2, 256, 128, 256, 2, torch.float16, 64),     # fp16
        (1, 4, 512, 128, 512, 4, torch.bfloat16, 64),    # 4 heads, 50% density
        (1, 12, 768, 128, 768, 6, torch.bfloat16, 64),   # 12 heads
    ]
    for cfg in small_configs:
        r = test_vs_pytorch(*cfg)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.config}  rel_max={r.rel_max_diff:.6e}  avg={r.avg_abs_diff:.6e}")

    # ---- Test 2: Medium configs vs Triton ----
    print("\n[2/6] Medium configs vs Triton reference")
    # NOTE: Triton fwd hardcodes tl.bfloat16 cast — fp16 Triton tests skipped.
    # NOTE: block_m=128 uses different LUT Q-stride (ceil(Sq/128)) — tested separately via CK-only path.
    medium_configs = [
        (1, 12, 4096, 128, 4096, 6, torch.bfloat16, 64),
        (1, 12, 4096, 128, 4096, 32, torch.bfloat16, 64),   # 50% density
        (1, 12, 4096, 128, 4096, 64, torch.bfloat16, 64),   # 100% density
        (1, 24, 4096, 128, 4096, 6, torch.bfloat16, 64),    # 24 heads
    ]
    for cfg in medium_configs:
        r = test_vs_triton(*cfg)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.config}  rel_max={r.rel_max_diff:.6e}  avg={r.avg_abs_diff:.6e}")

    # ---- Test 3: Large configs vs Triton ----
    print("\n[3/6] Large configs vs Triton reference")
    large_configs = [
        (1, 12, 16384, 128, 16384, 25, torch.bfloat16, 64),
        (1, 12, 32768, 128, 32768, 51, torch.bfloat16, 64),
        (1, 12, 49152, 128, 49152, 76, torch.bfloat16, 64),
        (1, 12, 65536, 128, 65536, 102, torch.bfloat16, 64),
    ]
    for cfg in large_configs:
        r = test_vs_triton(*cfg)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.config}  rel_max={r.rel_max_diff:.6e}  avg={r.avg_abs_diff:.6e}")

    # ---- Test 4: Asymmetric Q/KV lengths ----
    print("\n[4/6] Asymmetric Q/KV lengths vs Triton")
    asym_configs = [
        (1, 12, 4096, 128, 8192, 6, torch.bfloat16, 64),    # Sk > Sq
        (1, 12, 8192, 128, 4096, 6, torch.bfloat16, 64),    # Sq > Sk
        (1, 12, 4096, 128, 16384, 10, torch.bfloat16, 64),
    ]
    for cfg in asym_configs:
        r = test_vs_triton(*cfg)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.config}  rel_max={r.rel_max_diff:.6e}  avg={r.avg_abs_diff:.6e}")

    # ---- Test 5: Edge cases ----
    print("\n[5/6] Edge cases")
    edge_configs = [
        (1, 1, 64, 128, 64, 1, torch.bfloat16, 64),     # single block Q and KV
        (1, 24, 4096, 128, 4096, 1, torch.bfloat16, 64), # topk=1 (very sparse)
        (1, 1, 4096, 128, 4096, 64, torch.bfloat16, 64), # 100% dense, 1 head
    ]
    for cfg in edge_configs:
        r = test_vs_triton(*cfg)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.config}  rel_max={r.rel_max_diff:.6e}  avg={r.avg_abs_diff:.6e}")

    # ---- Test 6: Statistical consistency (multiple seeds) ----
    print("\n[6/6] Statistical consistency (10 random seeds)")
    max_rels = []
    for seed in range(10):
        r = test_vs_triton(1, 12, 4096, 128, 4096, 6, torch.bfloat16, 64, seed=seed)
        max_rels.append(r.rel_max_diff)
    avg_rel = sum(max_rels) / len(max_rels)
    worst_rel = max(max_rels)
    passed = worst_rel < 0.005
    results.append(TestResult(
        name="statistical_10seeds", passed=passed,
        max_abs_diff=0, rel_max_diff=worst_rel, avg_abs_diff=avg_rel,
        config="10_seeds,Sq=4096,topk=6",
        note=f"avg_rel={avg_rel:.6e}, worst_rel={worst_rel:.6e}"))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] 10 seeds: avg_rel={avg_rel:.6e}, worst_rel={worst_rel:.6e}")

    return results


# ─────────────────────────── Benchmark ─────────────────────────────────────

def flops_sparse(B, H, D, Sq, topk, block_n=64):
    return 4.0 * B * H * D * Sq * (topk * block_n)


def run_benchmarks() -> List[BenchResult]:
    from triton.testing import do_bench

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK — CK vs Triton block-sparse attention forward")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"BLOCK_M/N: {BLOCK}")

    results = []

    bench_configs = [
        # (B, H, Sq, D, Sk, topk, dtype, block_m, label)
        (1, 12, 4096, 128, 4096, 6, torch.bfloat16, 64, "Small sparse bm=64"),
        (1, 12, 4096, 128, 4096, 6, torch.bfloat16, 128, "Small sparse bm=128"),
        (1, 12, 8192, 128, 8192, 12, torch.bfloat16, 64, "8K sparse bm=64"),
        (1, 12, 8192, 128, 8192, 12, torch.bfloat16, 128, "8K sparse bm=128"),
        (1, 12, 16384, 128, 16384, 25, torch.bfloat16, 64, "16K sparse bm=64"),
        (1, 12, 16384, 128, 16384, 25, torch.bfloat16, 128, "16K sparse bm=128"),
        (1, 12, 32768, 128, 32768, 51, torch.bfloat16, 64, "32K sparse bm=64"),
        (1, 12, 32768, 128, 32768, 51, torch.bfloat16, 128, "32K sparse bm=128"),
        (1, 12, 49152, 128, 49152, 76, torch.bfloat16, 64, "49K sparse bm=64"),
        (1, 12, 49152, 128, 49152, 76, torch.bfloat16, 128, "49K sparse bm=128"),
        (1, 12, 65536, 128, 65536, 102, torch.bfloat16, 64, "65K sparse bm=64"),
        (1, 12, 65536, 128, 65536, 102, torch.bfloat16, 128, "65K sparse bm=128"),
        # Sparsity sweep at 49K
        (1, 12, 49152, 128, 49152, 38, torch.bfloat16, 128, "49K 5% dense"),
        (1, 12, 49152, 128, 49152, 153, torch.bfloat16, 128, "49K 20% dense"),
        (1, 12, 49152, 128, 49152, 384, torch.bfloat16, 128, "49K 50% dense"),
        (1, 12, 49152, 128, 49152, 768, torch.bfloat16, 128, "49K 100% dense"),
        # Asymmetric
        (1, 12, 49152, 128, 16384, 25, torch.bfloat16, 128, "49K→16K asym bm=128"),
    ]

    hdr = f"{'Label':<22} {'Sq':>6} {'Sk':>6} {'topk':>5} {'bm':>3} {'dtype':>5} {'Triton ms':>10} {'CK ms':>10} {'Speedup':>8} {'CK TFLOPS':>10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for B, H, Sq, D, Sk, topk, dtype, block_m, label in bench_configs:
        q, k, v, bm, q2k_idx, q2k_num, vbs = make_inputs(B, H, Sq, D, Sk, topk, dtype)

        ms_tri = do_bench(
            lambda: triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs),
            warmup=5, rep=30)

        ms_ck = do_bench(
            lambda: ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, block_m),
            warmup=5, rep=30)

        speedup = ms_tri / ms_ck
        fl = flops_sparse(B, H, D, Sq, topk)
        ck_tflops = fl / ms_ck * 1e-12 * 1e3

        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
        r = BenchResult(config=label, Sq=Sq, Sk=Sk, topk=topk, heads=H,
                        head_dim=D, dtype=dtype_str, block_m=block_m,
                        triton_ms=round(ms_tri, 3), ck_ms=round(ms_ck, 3),
                        speedup=round(speedup, 2), ck_tflops=round(ck_tflops, 1))
        results.append(r)

        print(f"{label:<22} {Sq:6d} {Sk:6d} {topk:5d} {block_m:3d} {dtype_str:>5} "
              f"{ms_tri:10.3f} {ms_ck:10.3f} {speedup:8.2f}x {ck_tflops:10.1f}")

    return results


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="correctness only, skip benchmarks")
    parser.add_argument("--bench-only", action="store_true", help="benchmarks only, skip correctness")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"BLOCK size: {BLOCK}")

    correctness_results = []
    bench_results = []

    if not args.bench_only:
        correctness_results = run_correctness_tests()

        # Summary
        n_pass = sum(1 for r in correctness_results if r.passed)
        n_total = len(correctness_results)
        print(f"\n{'=' * 80}")
        print(f"CORRECTNESS SUMMARY: {n_pass}/{n_total} PASSED")
        if n_pass < n_total:
            print("FAILURES:")
            for r in correctness_results:
                if not r.passed:
                    print(f"  {r.name}: {r.note}")
        print(f"{'=' * 80}")

    if not args.quick:
        bench_results = run_benchmarks()

        # Summary
        print(f"\n{'=' * 80}")
        print("BENCHMARK SUMMARY")
        if bench_results:
            speedups = [r.speedup for r in bench_results]
            print(f"  Min speedup: {min(speedups):.2f}x")
            print(f"  Max speedup: {max(speedups):.2f}x")
            print(f"  Avg speedup: {sum(speedups)/len(speedups):.2f}x")
            best = max(bench_results, key=lambda r: r.ck_tflops)
            print(f"  Peak CK TFLOPs: {best.ck_tflops:.1f} ({best.config})")
        print(f"{'=' * 80}")

    # Output JSON
    if args.output:
        out = {
            "device": device,
            "torch_version": torch.__version__,
            "block_size": BLOCK,
            "correctness": [asdict(r) for r in correctness_results],
            "benchmarks": [asdict(r) for r in bench_results],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
