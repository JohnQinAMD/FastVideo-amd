#!/usr/bin/env python3
"""
Strict correctness + performance test for CK VSA block-sparse attention forward.

Three-way comparison:
  - PyTorch fp32 reference (ground truth)
  - Triton bf16 (existing)
  - CK bf16 (new)

Metrics:
  - Cosine similarity (most robust for attention outputs)
  - RMSE / mean-normalized RMSE
  - Max absolute error
  - 99th percentile absolute error

Usage:
  python3 strict_test_ck_fwd.py              # full suite
  python3 strict_test_ck_fwd.py --quick      # correctness only
  python3 strict_test_ck_fwd.py --bench-only # benchmarks only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict

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

# Import CK extension directly (no package __init__.py dependency)
import ck_vsa_ops

# Import submodules directly to avoid fastvideo_kernel.__init__ pulling in
# ops/vmoba/turbodiffusion (which require compiled C++ extensions).
import importlib, importlib.util

def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "python", "fastvideo_kernel")
# Triton kernels have no external deps beyond torch/triton
_triton_index = _import_from_file(
    "fastvideo_kernel.triton_kernels.index",
    os.path.join(_pkg_dir, "triton_kernels", "index.py"))
# Stub the package so block_sparse_attn's lazy import of triton_kernels.index works
sys.modules.setdefault("fastvideo_kernel", type(sys)("fastvideo_kernel"))
sys.modules.setdefault("fastvideo_kernel.triton_kernels", type(sys)("fastvideo_kernel.triton_kernels"))
sys.modules["fastvideo_kernel.triton_kernels"].index = _triton_index
sys.modules["fastvideo_kernel.triton_kernels.index"] = _triton_index

_block_sparse = _import_from_file(
    "fastvideo_kernel.block_sparse_attn",
    os.path.join(_pkg_dir, "block_sparse_attn.py"))
_map_to_index = _block_sparse._map_to_index

_triton_attn = _import_from_file(
    "fastvideo_kernel.triton_kernels.block_sparse_attn_triton",
    os.path.join(_pkg_dir, "triton_kernels", "block_sparse_attn_triton.py"))
triton_block_sparse_attn_forward = _triton_attn.triton_block_sparse_attn_forward

BLOCK = 64


# ─────────────────────────── Helpers ───────────────────────────────────────

def pytorch_reference(q, k, v, block_map, variable_block_sizes):
    """Dense PyTorch fp32 reference: masked attention."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    nq = Sq // BLOCK
    nkv = Sk // BLOCK

    full_mask = torch.zeros(B, H, Sq, Sk, dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for qi in range(nq):
                for ki in range(nkv):
                    if block_map[b, h, qi, ki]:
                        bs = variable_block_sizes[ki].item()
                        full_mask[b, h, qi*BLOCK:(qi+1)*BLOCK, ki*BLOCK:ki*BLOCK+bs] = True

    q_f, k_f, v_f = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(D)
    qk = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    qk = qk.masked_fill(~full_mask, float('-inf'))
    attn = torch.nn.functional.softmax(qk, dim=-1)
    attn = attn.masked_fill(torch.isnan(attn), 0.0)
    return torch.matmul(attn, v_f)  # fp32 output


def make_inputs(B, H, Sq, D, Sk, topk, seed=42, partial_blocks=False):
    """Create bf16 inputs. If partial_blocks, some KV blocks have < 64 tokens."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device="cuda")

    nq, nkv = Sq // BLOCK, Sk // BLOCK
    topk = min(max(1, topk), nkv)

    scores = torch.rand(B, H, nq, nkv, device="cuda")
    idx = torch.topk(scores, topk, dim=-1).indices
    block_map = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
    block_map.scatter_(-1, idx, True)

    q2k_idx, q2k_num = _map_to_index(block_map)

    if partial_blocks:
        # Simulate FastVideo's 3D tiling: last block in each dim may be partial.
        # Make ~25% of blocks have 16-48 tokens, rest have 64.
        vbs = torch.full((nkv,), BLOCK, dtype=torch.int32, device="cuda")
        torch.manual_seed(seed + 999)
        partial_mask = torch.rand(nkv, device="cuda") < 0.25
        partial_sizes = torch.randint(16, 49, (nkv,), device="cuda", dtype=torch.int32)
        vbs[partial_mask] = partial_sizes[partial_mask]
        # Zero out K/V at padded positions (simulates vsa_pad behavior)
        for ki in range(nkv):
            bs = vbs[ki].item()
            if bs < BLOCK:
                k[:, :, ki*BLOCK + bs : (ki+1)*BLOCK, :] = 0
                v[:, :, ki*BLOCK + bs : (ki+1)*BLOCK, :] = 0
    else:
        vbs = torch.full((nkv,), BLOCK, dtype=torch.int32, device="cuda")

    return q, k, v, block_map, q2k_idx, q2k_num, vbs


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy metrics between ref (fp32) and test."""
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    diff = (ref_f - test_f).abs()

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()

    # RMSE and normalized RMSE
    rmse = diff.pow(2).mean().sqrt().item()
    ref_rms = ref_f.pow(2).mean().sqrt().item()
    nrmse = rmse / max(ref_rms, 1e-10)

    # quantile() OOMs on large tensors; use kth_value on a sample instead
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    n = diff.numel()
    if n > 10_000_000:
        idx = torch.randperm(n, device=diff.device)[:1_000_000]
        p99 = diff.flatten()[idx].quantile(0.99).item()
    else:
        p99 = diff.quantile(0.99).item()

    return {
        "cosine_sim": cos,
        "rmse": rmse,
        "nrmse": nrmse,
        "max_abs_err": max_abs,
        "p99_abs_err": p99,
        "mean_abs_err": mean_abs,
    }


@dataclass
class CorrectnessResult:
    config: str
    triton_metrics: Dict[str, float]
    ck_metrics: Dict[str, float]
    ck_vs_triton_metrics: Dict[str, float]
    passed: bool
    note: str = ""


@dataclass
class BenchResult:
    config: str
    Sq: int
    Sk: int
    topk: int
    density_pct: float
    triton_ms: float
    ck_ms: float
    speedup: float
    ck_tflops: float


# ─────────────────────────── Correctness ───────────────────────────────────

def run_three_way_test(B, H, Sq, D, Sk, topk, seed=42, skip_pytorch_ref=False,
                       partial_blocks=False) -> CorrectnessResult:
    """Three-way comparison: PyTorch fp32 vs Triton bf16 vs CK bf16."""
    q, k, v, bm, q2k_idx, q2k_num, vbs = make_inputs(
        B, H, Sq, D, Sk, topk, seed, partial_blocks=partial_blocks)

    # Ground truth (skip for large seqlens to avoid OOM)
    ref = None
    if not skip_pytorch_ref:
        ref = pytorch_reference(q, k, v, bm, vbs)

    # Triton (only for D=128 — Triton hardcodes bf16 MFMA and uses BLOCK=64 fixed)
    o_tri = None
    if D == 128:
        o_tri, _ = triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs)
        torch.cuda.synchronize()

    # CK
    o_ck, _ = ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, vbs, 64)
    torch.cuda.synchronize()

    triton_m = compute_metrics(ref, o_tri) if (ref is not None and o_tri is not None) else None
    ck_m = compute_metrics(ref, o_ck) if ref is not None else None
    ck_vs_tri = compute_metrics(o_tri.float(), o_ck) if o_tri is not None else None

    nkv = Sk // BLOCK
    density = topk / nkv * 100

    cfg = f"B={B} H={H} Sq={Sq} Sk={Sk} D={D} topk={topk} ({density:.0f}%)"

    # Pass criteria:
    # - If Triton available: CK vs Triton cosine >= 0.99999
    # - If fp32 ref available: CK vs fp32 cosine >= 0.9999
    # - If only CK vs fp32 (no Triton, e.g. D=64): CK vs fp32 cosine >= 0.9999
    passed = True
    note = ""
    if ck_vs_tri is not None:
        passed = passed and ck_vs_tri["cosine_sim"] >= 0.99999
    if ck_m is not None:
        passed = passed and ck_m["cosine_sim"] >= 0.9999
    if not passed:
        parts = []
        if ck_vs_tri: parts.append(f"CK-Tri cos={ck_vs_tri['cosine_sim']:.6f}")
        if ck_m: parts.append(f"CK-ref cos={ck_m['cosine_sim']:.6f}")
        note = " ".join(parts)

    return CorrectnessResult(
        config=cfg,
        triton_metrics=triton_m,
        ck_metrics=ck_m,
        ck_vs_triton_metrics=ck_vs_tri,
        passed=passed,
        note=note,
    )


def run_correctness_tests() -> List[CorrectnessResult]:
    results = []

    print("=" * 100)
    print("CORRECTNESS: Three-way comparison (PyTorch fp32 vs Triton bf16 vs CK bf16)")
    print("=" * 100)
    print()
    print("Note: Triton forward hardcodes bf16 MFMA (p.to(tl.bfloat16) at line 153).")
    print("      Both Triton and CK use online softmax in bf16 → expect ~1-3% max pointwise")
    print("      deviation from fp32 reference. Cosine similarity is the primary metric.")
    print()

    configs = [
        # (B, H, Sq, D, Sk, topk)
        # Small
        (1, 1, 128, 128, 128, 1),
        (1, 1, 128, 128, 128, 2),
        (1, 2, 256, 128, 256, 2),
        (1, 4, 512, 128, 512, 4),
        (1, 12, 768, 128, 768, 6),
        # Medium
        (1, 12, 4096, 128, 4096, 6),
        (1, 12, 4096, 128, 4096, 32),
        (1, 12, 4096, 128, 4096, 64),
        (1, 24, 4096, 128, 4096, 6),
        # Large
        (1, 12, 16384, 128, 16384, 25),
        (1, 12, 32768, 128, 32768, 51),
        (1, 12, 49152, 128, 49152, 76),
        (1, 12, 65536, 128, 65536, 102),
        # Asymmetric
        (1, 12, 4096, 128, 8192, 6),
        (1, 12, 8192, 128, 4096, 6),
        # Edge cases
        (1, 1, 64, 128, 64, 1),
        (1, 24, 4096, 128, 4096, 1),
        (1, 1, 4096, 128, 4096, 64),
        # Head dim = 64 (CK vs fp32 only — Triton hardcodes D=128 MFMA)
        (1, 12, 4096, 64, 4096, 6),
        (1, 12, 4096, 64, 4096, 32),
    ]

    hdr = (f"{'Config':<48} {'':>3} {'Cos Sim':>9} {'NRMSE':>9} {'MaxAbs':>9} "
           f"{'P99Abs':>9} {'MeanAbs':>10}")
    print(hdr)
    print("-" * len(hdr))

    for cfg in configs:
        # Skip fp32 reference for large seqlens (OOM: full QK matrix > 100 GB)
        max_sq = max(cfg[2], cfg[4])  # Sq or Sk
        skip_ref = (max_sq * cfg[1]) > 16384 * 12  # heuristic for ~200K tokens * heads
        r = run_three_way_test(*cfg, skip_pytorch_ref=skip_ref)
        results.append(r)
        status = "OK" if r.passed else "!!"
        def fmt_line(label, m):
            return (f"{'':48} {label} {m['cosine_sim']:9.6f} {m['nrmse']:9.5f} "
                    f"{m['max_abs_err']:9.4f} {m['p99_abs_err']:9.4f} {m['mean_abs_err']:10.2e}")
        if r.triton_metrics:
            print(f"{r.config:<48}" + fmt_line("Tri", r.triton_metrics)[48:])
        if r.ck_metrics:
            print(fmt_line("CK ", r.ck_metrics) + f"  [{status}]")
        if r.ck_vs_triton_metrics:
            print(fmt_line("C-T", r.ck_vs_triton_metrics) +
                  (f"  [{status}]" if r.ck_metrics is None else ""))
        elif r.ck_metrics is None:
            print(f"{r.config:<48} (CK vs fp32 only)  [{status}]")
        print()

    # Variable block sizes (partial blocks, simulates FastVideo 3D tiling edge cases)
    print("\n--- Variable block sizes (partial blocks) ---")
    vbs_configs = [
        (1, 4, 512, 128, 512, 4),
        (1, 12, 4096, 128, 4096, 6),
        (1, 12, 4096, 128, 4096, 32),
    ]
    for cfg in vbs_configs:
        r = run_three_way_test(*cfg, partial_blocks=True)
        results.append(r)
        status = "OK" if r.passed else "!!"
        cm = r.ck_metrics
        if cm:
            print(f"  [{status}] B={cfg[0]} H={cfg[1]} Sq={cfg[2]} D={cfg[3]} topk={cfg[5]} (partial)"
                  f"  CK-ref cos={cm['cosine_sim']:.6f}  nrmse={cm['nrmse']:.5f}")

    # Multi-seed stability
    print("\n--- Multi-seed stability (10 seeds, Sq=4096, H=12, topk=6) ---")
    seed_results = []
    for seed in range(10):
        r = run_three_way_test(1, 12, 4096, 128, 4096, 6, seed=seed)
        seed_results.append(r)
    tri_cos = [r.triton_metrics["cosine_sim"] for r in seed_results if r.triton_metrics]
    ck_cos = [r.ck_metrics["cosine_sim"] for r in seed_results if r.ck_metrics]
    ct_cos = [r.ck_vs_triton_metrics["cosine_sim"] for r in seed_results if r.ck_vs_triton_metrics]
    if tri_cos:
        print(f"  Triton vs fp32:  cos_sim min={min(tri_cos):.6f}  avg={sum(tri_cos)/len(tri_cos):.6f}")
    if ck_cos:
        print(f"  CK vs fp32:     cos_sim min={min(ck_cos):.6f}  avg={sum(ck_cos)/len(ck_cos):.6f}")
    if ct_cos:
        print(f"  CK vs Triton:   cos_sim min={min(ct_cos):.6f}  avg={sum(ct_cos)/len(ct_cos):.6f}")

    return results


# ─────────────────────────── Benchmark ─────────────────────────────────────

def flops_sparse(B, H, D, Sq, topk, block_n=64):
    return 4.0 * B * H * D * Sq * (topk * block_n)


def run_benchmarks() -> List[BenchResult]:
    from triton.testing import do_bench

    print()
    print("=" * 100)
    print("PERFORMANCE: CK vs Triton block-sparse attention forward (bf16, block_m=64)")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: batch=1, heads=12, head_dim=128, block_size=64")
    print()

    results = []

    bench_configs = [
        # (B, H, Sq, D, Sk, topk)
        # Seq length scaling (~10% density)
        (1, 12, 4096, 128, 4096, 6),
        (1, 12, 8192, 128, 8192, 12),
        (1, 12, 16384, 128, 16384, 25),
        (1, 12, 32768, 128, 32768, 51),
        (1, 12, 49152, 128, 49152, 76),
        (1, 12, 65536, 128, 65536, 102),
        # Sparsity sweep at 49K
        (1, 12, 49152, 128, 49152, 38),
        (1, 12, 49152, 128, 49152, 153),
        (1, 12, 49152, 128, 49152, 384),
        (1, 12, 49152, 128, 49152, 768),
        # Asymmetric
        (1, 12, 49152, 128, 16384, 25),
    ]

    hdr = f"{'Sq':>6} {'Sk':>6} {'topk':>5} {'density':>7} {'Triton ms':>10} {'CK ms':>10} {'Speedup':>8} {'CK TFLOPS':>10}"
    print(hdr)
    print("-" * len(hdr))

    for B, H, Sq, D, Sk, topk in bench_configs:
        q, k, v, bm, q2k_idx, q2k_num, vbs = make_inputs(B, H, Sq, D, Sk, topk)
        nkv = Sk // BLOCK
        density = topk / nkv * 100

        ms_tri = do_bench(
            lambda: triton_block_sparse_attn_forward(q, k, v, q2k_idx, q2k_num, vbs),
            warmup=5, rep=30)

        ms_ck = do_bench(
            lambda: ck_vsa_ops.ck_block_sparse_attn_fwd(q, k, v, q2k_idx, q2k_num, vbs, 64),
            warmup=5, rep=30)

        speedup = ms_tri / ms_ck
        fl = flops_sparse(B, H, D, Sq, topk)
        ck_tflops = fl / ms_ck * 1e-12 * 1e3

        r = BenchResult(config=f"Sq={Sq},Sk={Sk},topk={topk}",
                        Sq=Sq, Sk=Sk, topk=topk, density_pct=round(density, 1),
                        triton_ms=round(ms_tri, 3), ck_ms=round(ms_ck, 3),
                        speedup=round(speedup, 2), ck_tflops=round(ck_tflops, 1))
        results.append(r)

        print(f"{Sq:6d} {Sk:6d} {topk:5d} {density:6.1f}% {ms_tri:10.3f} {ms_ck:10.3f} "
              f"{speedup:8.2f}x {ck_tflops:10.1f}")

    return results


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    correctness_results = []
    bench_results = []

    if not args.bench_only:
        correctness_results = run_correctness_tests()
        n_pass = sum(1 for r in correctness_results if r.passed)
        n_total = len(correctness_results)
        print(f"\nCORRECTNESS SUMMARY: {n_pass}/{n_total} PASSED")
        if n_pass < n_total:
            for r in correctness_results:
                if not r.passed:
                    print(f"  FAIL: {r.config} — {r.note}")

    if not args.quick:
        bench_results = run_benchmarks()
        if bench_results:
            speedups = [r.speedup for r in bench_results]
            print(f"\nBENCHMARK SUMMARY: {min(speedups):.2f}x – {max(speedups):.2f}x "
                  f"(avg {sum(speedups)/len(speedups):.2f}x)")

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
        print(f"\nJSON: {args.output}")


if __name__ == "__main__":
    main()
