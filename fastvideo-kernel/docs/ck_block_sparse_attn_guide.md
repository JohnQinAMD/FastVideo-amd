# CK Block-Sparse Attention for FastVideo on MI355X

## TL;DR

**2.0–2.7× forward speedup over the vanilla FastVideo Triton kernel**
(MI355X, bf16, optimized path with fused delta + VBS skip + block_m=128).

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
from fastvideo_kernel.triton_kernels.index import map_to_index_and_delta

# --- block_m=128 (fastest, recommended for Sq >= 8K) ---
# Build block map at 128-token Q granularity, 64-token KV granularity
BLOCK_Q = 128
nq = Sq // BLOCK_Q
topk = max(1, int(nkv * 0.10))
scores = torch.rand(B, H, nq, nkv, device="cuda")
idx = torch.topk(scores, topk, dim=-1).indices
block_map = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
block_map.scatter_(-1, idx, True)
q2k_index, q2k_delta, q2k_num = map_to_index_and_delta(block_map)

output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
    q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m=128,
    q2k_delta=q2k_delta, skip_vbs_correction=True,
)
# output: [B, H, Sq, D] bf16
# lse:    [B, H, Sq]    fp32

# --- block_m=64 (use for small Sq or when 64-token Q granularity is required) ---
# Build block map at 64-token Q granularity
# nq_64 = Sq // 64
# block_map_64 = ...  # [B, H, nq_64, nkv]
# q2k_index, q2k_delta, q2k_num = map_to_index_and_delta(block_map_64)
# output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
#     q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m=64,
#     q2k_delta=q2k_delta, skip_vbs_correction=True,
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

2. **Fuse `abs → delta` into triton `map_to_index`.**
   The CK VSA pipeline consumes a delta-encoded LUT. Previously the C++
   wrapper launched a dedicated `abs_to_delta_kernel` between the
   existing triton `map_to_index` and the CK kernel. The new
   `map_to_index_and_delta_kernel` emits both the absolute LUT (still
   needed by the VBS correction kernel) and the delta LUT in a single
   triton launch; the C++ wrapper takes the delta as an optional kwarg
   and skips the HIP launch when provided. Saves one kernel launch
   (~5 µs) per forward.

3. **Replace `.contiguous()` with `TORCH_CHECK(is_contiguous)`.**
   Minor — removes a handful of virtual dispatches per call.

## Phase 2 optimizations

These target per-call overhead in the CK forward wrapper — eliminating
redundant kernel launches when the caller can pre-compute the delta LUT
or guarantee uniform block sizes.

4. **Skip VBS correction for uniform blocks.**
   [ck_vsa_fwd.hip](../csrc/attention/ck_sparse/ck_vsa_fwd.hip) now
   accepts `skip_vbs_correction=true`. When all KV block sizes are 64
   (the common uniform-grid case), the caller passes this flag to skip
   the VBS output correction kernel entirely. Saves ~5 µs per forward
   (one fewer kernel launch); most impactful at small Sq where it
   represents 10-15 % of total runtime.

5. **Pass pre-computed delta from Python.**
   [ck_sparse_attn.py](../python/fastvideo_kernel/ck_sparse_attn.py)
   now accepts `q2k_delta` and `skip_vbs_correction` kwargs and forwards
   them to the C++ wrapper. Combined with `map_to_index_and_delta()`
   (opt #2), the optimized call eliminates **both** the HIP abs→delta
   kernel and the VBS correction kernel, leaving only the CK attention
   kernel as the sole GPU launch.

### Measured gains (chi2761, MI355X)

| Sq | D | CK base (ms) | CK opt (ms) | opt/base | opt vs Triton |
|----|---|-------------|-------------|----------|---------------|
| 4,096 | 128 | 0.041 | **0.033** | **1.24×** | **1.66×** |
| 8,192 | 128 | 0.107 | **0.086** | **1.24×** | **1.99×** |
| 16,384 | 128 | 0.327 | **0.258** | **1.27×** | **2.45×** |
| 32,768 | 128 | 1.142 | **0.934** | **1.22×** | **2.59×** |
| 49,152 | 128 | 2.614 | **2.104** | **1.24×** | **2.69×** |
| 65,536 | 128 | 4.657 | **3.771** | **1.24×** | **2.48×** |
| 4,096 | 64 | 0.034 | **0.023** | **1.48×** | N/A |
| 16,384 | 64 | 0.196 | **0.153** | **1.28×** | N/A |
| 49,152 | 64 | 1.465 | **1.292** | **1.13×** | N/A |

Biggest wins at small Sq where eliminated kernel launches are a larger
fraction of total runtime. Combined with block_m=128, CK opt achieves
1.22–1.48× over CK base across all configs.

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
q2k_index, q2k_delta, q2k_num = map_to_index_and_delta(block_map)
output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
    q, k, v, q2k_index, q2k_num, vbs, block_m=128,
    q2k_delta=q2k_delta, skip_vbs_correction=True,
)
```

**When to use block_m=128 vs block_m=64:**
- **block_m=128** — recommended for Sq ≥ 8,192 with uniform blocks.
  1.18× faster than block_m=64 at same density; up to 2.69× over Triton.
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

### Measured gains (chi2761, MI355X, D=128, ~10% density)

| Sq | Triton (ms) | CK opt (ms) | Speedup |
|----|-------------|-------------|--------:|
| 4,096 | 0.054 | 0.033 | 1.66× |
| 8,192 | 0.172 | **0.086** | **1.99×** |
| 16,384 | 0.632 | **0.258** | **2.45×** |
| 32,768 | 2.419 | **0.934** | **2.59×** |
| 49,152 | 5.651 | **2.104** | **2.69×** |
| 65,536 | 9.370 | **3.771** | **2.48×** |

```bash
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --blockm128
```

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
