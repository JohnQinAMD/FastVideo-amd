# CK Block-Sparse Attention for FastVideo on MI355X

## TL;DR

**2.0–2.7× forward speedup over the vanilla FastVideo Triton kernel**
(MI355X, bf16, optimized path with VBS skip + block_m=128).

| Config | FV Triton | CK opt | Speedup |
|--------|-----------|--------|---------|
| D=128, Sq=4,096,  10% sparse | 0.054 ms | **0.033 ms** | **1.66×** |
| D=128, Sq=8,192,  10% sparse | 0.172 ms | **0.086 ms** | **1.99×** |
| D=128, Sq=16,384, 10% sparse | 0.632 ms | **0.258 ms** | **2.45×** |
| D=128, Sq=32,768, 10% sparse | 2.419 ms | **0.934 ms** | **2.59×** |
| D=128, Sq=49,152, 10% sparse | 5.651 ms | **2.104 ms** | **2.69×** |
| D=128, Sq=65,536, 10% sparse | 9.370 ms | **3.771 ms** | **2.48×** |

Workload: `B=1, H=12, block_size=64, topk=~10%`, bf16, forward-only.

Correctness: CK vs Triton cosine similarity = 1.000000 on all 23 test configs.
CK block_m=128 vs PyTorch fp32: cosine similarity ≥ 0.999997 on 23/23 configs.
The fused-delta path is bit-exact with the legacy in-wrapper delta path.

**To reproduce**, clone these branches:

| Repo | Branch | Content |
|------|--------|---------|
| [FastVideo-amd](https://github.com/JohnQinAMD/FastVideo-amd) | `feature/ck-block-sparse-attn-clean` | CK kernel wrapper, build script, test suite |
| [aiter-amd](https://github.com/JohnQinAMD/aiter-amd) | `sla-ck-fwd-release-inference` | CK headers with VSA sparse attention (submodule `composable_kernel` at `042d4f0`) |

---

## Requirements

- MI355X (gfx950) GPU
- `rocm/primus:v26.2` container image

## 1 — Setup

```bash
mkdir -p fastvideo-ck && cd fastvideo-ck

docker run -it --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --ipc host --network host --shm-size 64g \
    -v "$(pwd):/work" \
    rocm/primus:v26.2 bash
```

All remaining steps run inside the container.

## 2 — Clone

```bash
cd /work

git clone -b feature/ck-block-sparse-attn-clean \
    https://github.com/JohnQinAMD/FastVideo-amd.git

git clone --recurse-submodules --shallow-submodules \
    -b sla-ck-fwd-release-inference \
    https://github.com/JohnQinAMD/aiter-amd.git
```

## 3 — Build

```bash
cd /work/FastVideo-amd/fastvideo-kernel/csrc/attention/ck_sparse

export CK_DIR=/work/aiter-amd/3rdparty/composable_kernel
export GPU_ARCH=gfx950

./build.sh
```

The build patches the CK codegen for head_dim=64, generates kernel instances, and compiles (~2 min). Output: `build/ck_vsa_ops.cpython-3XX-x86_64-linux-gnu.so`.

## 4 — Correctness

```bash
cd /work/FastVideo-amd
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --quick
```

Expected: `CORRECTNESS SUMMARY: 23/23 PASSED`

Three-way validation: PyTorch fp32 (ground truth) vs Triton bf16 vs CK bf16.
CK vs Triton cosine similarity = 1.000000 on all configs.

## 5 — Performance

```bash
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --bench-only
```

### D=128, ~10% sparsity (MI355X, chi2761)

| Sq | FV Triton (ms) | CK base (ms) | CK opt (ms) | Speedup (vs Triton) |
|----|----------------|-------------|-------------|---------------------|
| 4,096 | 0.054 | 0.041 | **0.033** | **1.66×** |
| 8,192 | 0.172 | 0.107 | **0.086** | **1.99×** |
| 16,384 | 0.632 | 0.327 | **0.258** | **2.45×** |
| 32,768 | 2.419 | 1.142 | **0.934** | **2.59×** |
| **49,152** | **5.651** | **2.614** | **2.104** | **2.69×** |
| 65,536 | 9.370 | 4.657 | **3.771** | **2.48×** |

### D=64, ~10% sparsity

| Sq | CK base (ms) | CK opt (ms) | Speedup (opt/base) |
|----|-------------|-------------|-------------------:|
| 4,096 | 0.034 | **0.023** | **1.48×** |
| 16,384 | 0.196 | **0.153** | **1.28×** |
| **49,152** | **1.465** | **1.292** | **1.13×** |

Note: Triton forward hardcodes D=128 MFMA, so no Triton baseline for D=64.

### D=128, variable block sizes (~25% partial blocks)

| Sq | Triton (ms) | CK opt (ms) | Speedup |
|----|-------------|-------------|---------|
| 4,096 | 0.054 | 0.048 | 1.13x |
| 16,384 | 0.631 | 0.410 | 1.54x |
| 32,768 | 2.417 | 1.372 | 1.76x |
| **49,152** | **5.284** | **2.845** | **1.86x** |
| 65,536 | 9.370 | 4.932 | 1.90x |

Note: VBS workloads use block_m=64 (block_m=128 with partial blocks
requires merged Q-block rows, which doubles effective topk and is slower).

### D=128, sparsity sweep (Sq = 49,152)

| Density | Triton (ms) | CK opt (ms) | Speedup |
|---------|-------------|-------------|--------:|
| 5% | 2.75 | **1.11** | **2.49×** |
| 10% | 5.30 | **2.10** | **2.52×** |
| 20% | 10.45 | **4.14** | **2.53×** |
| 50% | 25.89 | **10.44** | **2.48×** |
| 100% | 51.61 | **20.42** | **2.53×** |

## 6 — Usage

```python
import sys, os
sys.path.insert(0, "fastvideo-kernel/csrc/attention/ck_sparse/build")
import torch, ck_vsa_ops

# Inputs: B=1, H=12, Sq=49152, D=128, ~10% sparsity
B, H, Sq, D, BLOCK_KV = 1, 12, 49152, 128, 64
nkv = Sq // BLOCK_KV

q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
variable_block_sizes = torch.full((nkv,), BLOCK_KV, dtype=torch.int32, device="cuda")

os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"
from fastvideo_kernel.triton_kernels.index import map_to_index

# --- block_m=128 (fastest, recommended for Sq >= 8K) ---
# Build block map at 128-token Q granularity, 64-token KV granularity
BLOCK_Q = 128
nq = Sq // BLOCK_Q
topk = max(1, int(nkv * 0.10))
scores = torch.rand(B, H, nq, nkv, device="cuda")
idx = torch.topk(scores, topk, dim=-1).indices
block_map = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
block_map.scatter_(-1, idx, True)
q2k_index, q2k_num = map_to_index(block_map)

output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
    q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m=128,
    skip_vbs_correction=True,
)
# output: [B, H, Sq, D] bf16
# lse:    [B, H, Sq]    fp32

# --- block_m=64 (use for small Sq or when 64-token Q granularity is required) ---
# Build block map at 64-token Q granularity
# nq_64 = Sq // 64
# block_map_64 = ...  # [B, H, nq_64, nkv]
# q2k_index, q2k_num = map_to_index(block_map_64)
# output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
#     q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m=64,
#     skip_vbs_correction=True,
# )
```

The CK kernel supports `D=64` or `D=128`, `bf16` or `fp16`, symmetric or
asymmetric Q/KV lengths, variable block sizes (1–64 tokens per block),
and `block_m=64` or `block_m=128` Q-tile granularity.

## Wrapper optimizations

Improvements below target per-call overhead — no CK-tile kernel changes.

1. **Drop the VBS host sync.**
   [ck_vsa_fwd.hip](../csrc/attention/ck_sparse/ck_vsa_fwd.hip) used to call
   `variable_block_sizes.cpu()` to decide whether the VBS correction
   kernel needed to launch; that forces a device→host sync on every
   forward. The correction kernel already early-exits per-thread when
   `pad_count == 0`, so launching unconditionally is cheaper than the
   sync. Largest win at small Sq (−25 % at Sq=4,096).

2. **Replace `.contiguous()` with `TORCH_CHECK(is_contiguous)`.**
   Minor — removes a handful of virtual dispatches per call.

## Phase 2 optimizations

These target per-call overhead in the CK forward wrapper — eliminating
redundant kernel launches when the caller can guarantee uniform block
sizes.

3. **Skip VBS correction for uniform blocks.**
   [ck_vsa_fwd.hip](../csrc/attention/ck_sparse/ck_vsa_fwd.hip) now
   accepts `skip_vbs_correction=true`. When all KV block sizes are 64
   (the common uniform-grid case), the caller passes this flag to skip
   the VBS output correction kernel entirely. Saves ~5 µs per forward
   (one fewer kernel launch); most impactful at small Sq where it
   represents 10-15 % of total runtime. This is what the `opt/base`
   column below measures; it leaves the CK attention kernel as the sole
   GPU launch.

### A note on the LUT encoding

Earlier revisions of this branch passed a separately built,
delta-encoded LUT to CK and credited it for part of the gain above. The
CK VSA pipeline differences the **absolute** block indices inline, so
the delta LUT was never read: `map_to_index` output is passed straight
through as `lut_ptr`. The delta plumbing (`abs_to_delta_kernel`, the
`q2k_delta` kwarg, `map_to_index_and_delta`) has been removed, and the
gains above are attributable entirely to skipping the VBS correction.

### Measured gains (MI355X, block_m=64, ~10 % density)

| Sq | D | CK base (ms) | CK opt (ms) | opt/base | opt vs Triton |
|----|---|-------------|-------------|----------|---------------|
| 4,096 | 128 | 0.034 | **0.031** | **1.09×** | **1.76×** |
| 8,192 | 128 | 0.096 | **0.092** | **1.05×** | **1.87×** |
| 16,384 | 128 | 0.305 | **0.296** | **1.03×** | **2.13×** |
| 32,768 | 128 | 1.094 | **1.076** | **1.02×** | **2.24×** |
| 49,152 | 128 | 2.453 | **2.385** | **1.03×** | **2.21×** |
| 65,536 | 128 | 4.373 | **4.308** | **1.02×** | **2.18×** |
| 4,096 | 64 | 0.027 | **0.024** | **1.11×** | **1.22×** |
| 16,384 | 64 | 0.169 | **0.164** | **1.03×** | **1.81×** |
| 49,152 | 64 | 1.443 | **1.401** | **1.03×** | **1.77×** |

Skipping the correction removes a fixed-cost launch, so the win is
largest at small Sq (1.09–1.11× at Sq=4,096) and fades to ~1.02× once
the attention kernel dominates. An earlier version of this table
reported 1.22–1.48×; those figures were measured with `block_m=128` and
also credited the delta LUT described above, which the kernel never
read. For the `block_m=128` numbers see the Phase 3 table below.

Note that the `CK base` arm must pass `skip_vbs_correction=False`
explicitly. The C++ default is now `true`, so a call that omits the
argument lands in the `CK opt` configuration.

## Phase 3: block_m=128 tile

The CK codegen compiles a kM0=128 tile (4 warps, 128×64×32) alongside
the default kM0=64 tile (2 warps). `block_m=128` processes 128 Q
tokens per threadblock, doubling K/V reuse per Q token.

**Why it helps.** `rocprofv3` PMC counters show the kernel is
memory-bound (`WAIT_ANY/MFMA ≈ 9–10`), not compute-bound. The
bottleneck is HBM bandwidth for K/V loads. `block_m=128` amortizes
each K/V load over 2× more Q tokens, directly improving the
compute/memory ratio. It also doubles wave occupancy (8 vs 4 waves/CU)
because LDS usage is unchanged (K/V prefetch buffers don't depend on
kM0) while the workgroup doubles from 2 to 4 warps.

**Usage.** Build the block map at 128-token Q granularity (nq = Sq/128)
with 64-token KV blocks (nkv = Sk/64), and pass `block_m=128`:

```python
nq_128 = Sq // 128
nkv = Sk // 64
block_map = ...  # [B, H, nq_128, nkv] bool
q2k_index, q2k_num = map_to_index(block_map)
output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
    q, k, v, q2k_index, q2k_num, vbs, block_m=128,
    skip_vbs_correction=True,
)
```

**When to use block_m=128 vs block_m=64:**
- **block_m=128** — recommended for Sq ≥ 8,192 with uniform blocks.
  1.20–1.24× faster than block_m=64 at same density; up to 2.74× over
  Triton. At Sq=4,096 the two tiles are within 3 % of each other.
- **block_m=64** — use for small Sq (< 8K), variable block sizes
  (partial blocks), or when 64-token Q granularity is required.

### PMC counter comparison (Sq=49,152, D=128, chi2761)

| Counter | kM0=64 | kM0=128 |
|---------|--------|---------|
| Warps/WG | 2 | 4 |
| LDS/WG | 27,136 B | 27,136 B |
| Waves/CU | 4 | **8** |
| WAIT_ANY/MFMA | 9.74 | **9.12** |
| VMEM_INSTS/MFMA | ~7.4 | **0.29** |

### Measured gains (MI355X, D=128, ~10% density)

| Sq | Triton (ms) | CK m=64 (ms) | CK m=128 (ms) | m128/m64 | m128 vs Triton |
|----|-------------|--------------|---------------|---------:|---------------:|
| 4,096 | 0.054 | 0.032 | 0.031 | 1.03× | 1.76× |
| 8,192 | 0.172 | 0.091 | **0.076** | **1.20×** | **2.26×** |
| 16,384 | 0.632 | 0.303 | **0.247** | **1.23×** | **2.56×** |
| 32,768 | 2.423 | 1.094 | **0.885** | **1.24×** | **2.74×** |
| 49,152 | 5.300 | 2.419 | **1.982** | **1.22×** | **2.67×** |
| 65,536 | 9.387 | 4.343 | **3.500** | **1.24×** | **2.68×** |

```bash
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --blockm128
```

## The high-density (HD) build

`csrc/attention/ck_sparse/build_hd.sh` produces a second extension whose
pipeline is overridden by
`build_hd/include_override/pipeline/block_fmha_pipeline_qr_ks_vs_async_vsa.hpp`.
It was written when the stock CK codegen emitted a kK0=32 tile, to
introduce a wider kK0=64 QK chunk plus a second QK accumulator that lets
consecutive outer-K steps issue independently.

**It is disabled by default and should stay that way on gfx950.** Two
things changed underneath it:

- The kK0=64 tile now ships in the stock CK codegen, so
  `patch_codegen_hd.py` is a no-op — both builds already compile the
  same `128x64x64x128x32x128` tile. `build_hd.sh` says as much in its
  log: *"HD bk0=64 tile already present in upstream codegen"*.
- What remains is the dual accumulator, and it costs more than it
  returns. It raises the kernel's VGPR count from 96 to 104, which on
  gfx950 (512 registers per SIMD) drops occupancy from 5 waves to 4.
  Since the kernel is memory-latency-bound, losing a wave of latency
  hiding outweighs the extra instruction-level parallelism: measured
  1–14 % slower than the stock build across Sq ∈ {8K…64K} and
  density ∈ {10 %, 30 %, 50 %}. Rebuilding the override with the
  `s_setprio` hints but *without* the dual accumulator gives 88 VGPRs,
  5 waves, and stock performance to within noise.

Set `FASTVIDEO_KERNEL_CK_HD_THRESHOLD` to a density in [0, 1] to
re-enable dispatch to it, e.g. when evaluating on an architecture with
a larger register file.

## Supported Features

| Feature | Status |
|---------|--------|
| Forward pass, bf16 / fp16 | Yes |
| head_dim = 64 and 128 | Yes |
| block_m = 64 and 128 | Yes |
| Variable block sizes | Yes |
| Asymmetric Q/KV lengths | Yes |
| Backward pass | Not yet |
| Torch autograd integration | Not yet |
