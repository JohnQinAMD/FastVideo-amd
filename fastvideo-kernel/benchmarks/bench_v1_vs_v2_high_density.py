"""Side-by-side: v1 (stock CK bk0=32) vs v2 (HD bk0=64 + QK outer-K dual-acc)
on captured high-density block-sparse patterns (~40 % topk, Sq ~80–90 K).

Both call the same C++ entrypoint `ck_block_sparse_attn_fwd`; only the
.so loaded differs.

Usage:
    # default: assumes repo layout, patterns dir via env var
    PATTERN_DIR=/path/to/benchmark_package \\
        python3 fastvideo-kernel/benchmarks/bench_v1_vs_v2_high_density.py

    # explicit:
    python3 fastvideo-kernel/benchmarks/bench_v1_vs_v2_high_density.py \\
        --pattern-dir /path/to/benchmark_package \\
        --fv-root .

Each pattern .pt file should contain `q_shape`, `block_m`, `topk`,
`q2k_index`, `q2k_num`, `vbs` keys.

Reports avg / min ms across 10 warmup + 30 timed runs for:
  - FV-bundled Triton baseline
  - v1 stock CK
  - v2 HD bk0=64 + QK outer-K dual-acc (shipped)
"""
import argparse
import importlib.util
import os
import sys
import torch


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in zip(s, e)]


def run_pattern(pf, pattern_dir, ck_v1, ck_v2, fv_triton_fwd, _map_to_index):
    d = torch.load(os.path.join(pattern_dir, pf),
                   map_location="cpu", weights_only=True)
    B, H, Sq, D = d["q_shape"]
    block_m = int(d["block_m"])
    nkv = Sq // 64
    full_topk = int(d["topk"])
    density = full_topk / nkv

    print(f"\n=== {pf} ===")
    print(f"  shape: B={B} H={H} Sq={Sq} D={D}  "
          f"block_m={block_m}  density={density*100:.1f}%")

    torch.manual_seed(42)
    q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
    qi = d["q2k_index"].cuda()
    qn = d["q2k_num"].cuda()
    vbs = d["vbs"].cuda()

    v1_ms = _bench(lambda: ck_v1.ck_block_sparse_attn_fwd(
        q, k, v, qi, qn, vbs, block_m, skip_vbs_correction=True)[0])
    v1_avg, v1_min = sum(v1_ms) / len(v1_ms), min(v1_ms)

    v2_ms = _bench(lambda: ck_v2.ck_block_sparse_attn_fwd(
        q, k, v, qi, qn, vbs, block_m, skip_vbs_correction=True)[0])
    v2_avg, v2_min = sum(v2_ms) / len(v2_ms), min(v2_ms)

    # Triton needs block_m=64 q2k upsampled from block_m=128.
    nq128 = Sq // 128
    ar = torch.arange(qi.shape[-1], device=qi.device, dtype=torch.int32)
    valid = ar.view(1, 1, 1, -1) < qn.unsqueeze(-1)
    safe_idx = torch.where(valid, qi, torch.zeros_like(qi)).long()
    bm128 = torch.zeros(B, H, nq128, nkv, dtype=torch.bool, device="cuda")
    bm128.scatter_(-1, safe_idx, valid)
    bm64 = bm128.repeat_interleave(2, dim=2)
    qi_tri, qn_tri = _map_to_index(bm64)

    tri_ms = _bench(lambda: fv_triton_fwd(q, k, v, qi_tri, qn_tri, vbs))
    tri_avg, tri_min = sum(tri_ms) / len(tri_ms), min(tri_ms)

    print(f"  {'kernel':<22s} {'avg_ms':>10s} {'min_ms':>10s} "
          f"{'vs Triton':>12s} {'v2/v1':>10s}")
    print(f"  {'Triton (FV bundled)':<22s} {tri_avg:>10.2f} {tri_min:>10.2f}")
    print(f"  {'v1: stock CK':<22s} {v1_avg:>10.2f} {v1_min:>10.2f}  "
          f"{tri_avg/v1_avg:>10.2f}x")
    print(f"  {'v2: HD+QKdualacc':<22s} {v2_avg:>10.2f} {v2_min:>10.2f}  "
          f"{tri_avg/v2_avg:>10.2f}x  {v1_avg/v2_avg:>9.3f}x")

    del q, k, v
    torch.cuda.empty_cache()
    return {"file": pf, "Sq": Sq, "density": density,
            "tri": tri_avg, "v1": v1_avg, "v2": v2_avg}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pattern-dir", default=os.environ.get("PATTERN_DIR"),
                   help=".pt files (default: $PATTERN_DIR)")
    p.add_argument("--fv-root", default=None,
                   help="FastVideo root (default: 3 levels up from this script)")
    p.add_argument("--patterns", nargs="*",
                   default=["e2e_shape_84480.pt", "e2e_shape_92160.pt"])
    args = p.parse_args()

    if not args.pattern_dir:
        sys.exit("--pattern-dir or $PATTERN_DIR is required")
    if not args.fv_root:
        # bench script lives at fastvideo-kernel/benchmarks/, so fv-root is 2 up
        args.fv_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."))

    fvk = os.path.join(args.fv_root, "fastvideo-kernel")
    stock_dir = os.path.join(fvk, "csrc/attention/ck_sparse/build")
    hd_dir = os.path.join(fvk, "csrc/attention/ck_sparse/build_hd")
    for d, name in [(stock_dir, "build/ck_vsa_ops"),
                    (hd_dir, "build_hd/ck_vsa_ops_hd")]:
        if not os.path.isdir(d):
            sys.exit(f"missing {name} under {d}; run ./build.sh + ./build_hd.sh first")

    # v1 stock CK
    sys.path.insert(0, stock_dir)
    import ck_vsa_ops as ck_v1
    sys.path.pop(0)

    # v2 HD bk0=64 + QK outer-K dual-acc
    sys.path.insert(0, hd_dir)
    import ck_vsa_ops_hd as ck_v2
    sys.path.pop(0)

    # FV Triton + Triton index helper
    os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"
    sys.modules.setdefault(
        "fastvideo_kernel", type(sys)("fastvideo_kernel"))
    sys.modules.setdefault(
        "fastvideo_kernel.triton_kernels",
        type(sys)("fastvideo_kernel.triton_kernels"))
    tri_index = _load_module(
        "fastvideo_kernel.triton_kernels.index",
        os.path.join(fvk, "python/fastvideo_kernel/triton_kernels/index.py"))
    sys.modules["fastvideo_kernel.triton_kernels"].index = tri_index
    fv_triton = _load_module(
        "fastvideo_kernel.triton_kernels.block_sparse_attn_triton",
        os.path.join(fvk, "python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py"))
    fv_triton_fwd = fv_triton.triton_block_sparse_attn_forward

    # _map_to_index helper from block_sparse_attn.py
    bsa = _load_module(
        "fastvideo_kernel.block_sparse_attn",
        os.path.join(fvk, "python/fastvideo_kernel/block_sparse_attn.py"))
    _map_to_index = bsa._map_to_index

    rows = []
    for pf in args.patterns:
        try:
            rows.append(run_pattern(pf, args.pattern_dir,
                                    ck_v1, ck_v2, fv_triton_fwd,
                                    _map_to_index))
        except Exception as e:
            print(f"FAILED {pf}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Summary ===")
    print(f"{'pattern':<24s} {'density':>8s} {'Triton':>10s} "
          f"{'v1':>10s} {'v2':>10s} {'v1/Tri':>9s} {'v2/Tri':>9s} {'v2/v1':>8s}")
    for r in rows:
        print(f"{r['file']:<24s} {r['density']*100:>7.1f}% "
              f"{r['tri']:>10.2f} {r['v1']:>10.2f} {r['v2']:>10.2f} "
              f"{r['tri']/r['v1']:>8.2f}x {r['tri']/r['v2']:>8.2f}x "
              f"{r['v1']/r['v2']:>7.3f}x")


if __name__ == "__main__":
    main()
