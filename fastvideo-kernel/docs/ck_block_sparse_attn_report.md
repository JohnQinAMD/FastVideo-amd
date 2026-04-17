# CK-Tile Block-Sparse Attention for FastVideo on AMD Instinct MI355X

**Date:** 2025-04-17  
**Author:** John Qin (AMD)  
**Source:** [`feature/ck-block-sparse-attn`](https://github.com/JohnQinAMD/FastVideo-amd/tree/feature/ck-block-sparse-attn)

---

## Summary

We replaced FastVideo's Triton-based block-sparse attention forward kernel with an AMD CK-tile implementation optimized for MI355X (gfx950). The CK kernel delivers a consistent **2x speedup** while producing **numerically identical outputs** to the existing Triton kernel (cosine similarity = 1.000000 across all tested configurations).

| | |
|---|---|
| **Hardware** | AMD Instinct MI355X, ROCm 7.2 |
| **Container** | `rocm/primus:v26.2` (PyTorch 2.10, Triton 3.6) |
| **Speedup range** | 1.4x (4K tokens) — 2.3x (65K tokens) |
| **Primary operating point** (49K tokens, 10% density) | 5.31 ms → 2.63 ms **(2.02x)** |
| **Peak throughput** | 646 TFLOPs bf16 |
| **CK vs Triton cosine similarity** | 1.000000 (all 18 test configs) |
| **Correctness tests** | 18/18 passed |

---

## 1. Correctness

Both implementations use bf16 online-softmax flash attention. We validate accuracy against a PyTorch fp32 dense-attention reference (ground truth) and directly cross-validate CK against Triton.

### vs PyTorch fp32 Reference

| Config | Triton cos sim | CK cos sim | CK vs Triton cos sim |
|--------|---------------|------------|---------------------|
| H=1, Sq=128, topk=1 (50%) | 0.999998 | 0.999998 | 1.000000 |
| H=1, Sq=128, topk=2 (100%) | 0.999998 | 0.999998 | 1.000000 |
| H=2, Sq=256, topk=2 | 0.999998 | 0.999998 | 1.000000 |
| H=4, Sq=512, topk=4 | 0.999998 | 0.999998 | 1.000000 |
| H=12, Sq=768, topk=6 | 0.999998 | 0.999998 | 1.000000 |
| H=12, Sq=4096, topk=6 (9%) | 0.999998 | 0.999998 | 1.000000 |
| H=12, Sq=4096, topk=32 (50%) | 0.999997 | 0.999997 | 1.000000 |
| H=12, Sq=4096, topk=64 (100%) | 0.999997 | 0.999997 | 1.000000 |
| H=24, Sq=4096, topk=6 | 0.999998 | 0.999998 | 1.000000 |
| H=12, Sq=16384, topk=25 | 0.999997 | 0.999998 | 1.000000 |
| H=12, Sq=4096→Sk=8192 (asym) | 0.999998 | 0.999998 | 1.000000 |
| H=12, Sq=8192→Sk=4096 (asym) | 0.999998 | 0.999998 | 1.000000 |
| H=1, Sq=64, topk=1 (single block) | 0.999998 | 0.999998 | 1.000000 |
| H=24, Sq=4096, topk=1 (2%) | 0.999998 | 0.999998 | 1.000000 |
| H=1, Sq=4096, topk=64 (100%) | 0.999997 | 0.999997 | 1.000000 |

### CK vs Triton (large sequences, fp32 reference skipped due to memory)

| Config | CK vs Triton cos sim | CK vs Triton NRMSE |
|--------|---------------------|--------------------|
| H=12, Sq=32768, topk=51 | 1.000000 | 0.00005 |
| H=12, Sq=49152, topk=76 | 1.000000 | 0.00005 |
| H=12, Sq=65536, topk=102 | 1.000000 | 0.00005 |

### Multi-Seed Stability (10 seeds, H=12, Sq=4096, topk=6)

| | Cosine Similarity (min) | Cosine Similarity (avg) |
|---|---|---|
| Triton vs fp32 | 0.999997 | 0.999998 |
| CK vs fp32 | 0.999997 | 0.999998 |
| CK vs Triton | 1.000000 | 1.000000 |

**18/18 tests pass.** CK and Triton produce identical output distributions across all sequence lengths, sparsity levels, head counts, and random seeds.

---

## 2. Performance

All benchmarks: batch=1, heads=12, head_dim=128, bf16, block_size=64.

### Sequence Length Scaling (~10% sparsity)

| Seq Length | topk | Density | Triton (ms) | CK (ms) | Speedup | CK TFLOPs |
|-----------|------|---------|-------------|---------|---------|-----------|
| 4,096 | 6 | 9.4% | 0.055 | 0.039 | **1.42x** | 249 |
| 8,192 | 12 | 9.4% | 0.173 | 0.105 | **1.64x** | 368 |
| 16,384 | 25 | 9.8% | 0.631 | 0.325 | **1.94x** | 495 |
| 32,768 | 51 | 10.0% | 2.423 | 1.170 | **2.07x** | 562 |
| **49,152** | **76** | **9.9%** | **5.312** | **2.632** | **2.02x** | **558** |
| 65,536 | 102 | 10.0% | 9.385 | 4.640 | **2.02x** | 566 |

### Sparsity Sweep (Sq = 49,152)

| Density | topk | Triton (ms) | CK (ms) | Speedup | CK TFLOPs |
|---------|------|-------------|---------|---------|-----------|
| 4.9% | 38 | 2.772 | 1.366 | **2.03x** | 538 |
| 9.9% | 76 | 5.312 | 2.632 | **2.02x** | 558 |
| 19.9% | 153 | 10.482 | 5.096 | **2.06x** | 580 |
| 50.0% | 384 | 26.014 | 12.425 | **2.09x** | 597 |
| 100.0% | 768 | 51.764 | 22.981 | **2.25x** | 646 |

### Asymmetric Q/KV Lengths

| Config | Triton (ms) | CK (ms) | Speedup |
|--------|-------------|---------|---------|
| Q=49K, KV=16K, topk=25 | 1.884 | 0.847 | **2.22x** |

---

## 3. How to Reproduce

### Prerequisites

- **GPU:** AMD Instinct MI355X (gfx950)
- **Container:** `rocm/primus:v26.2` (includes ROCm 7.2, PyTorch 2.10, Triton 3.6)
- **Source code:**
  - FastVideo repo, branch `feature/ck-block-sparse-attn`:
    ```bash
    git clone https://github.com/JohnQinAMD/FastVideo-amd.git
    cd FastVideo-amd
    git checkout feature/ck-block-sparse-attn
    ```
  - Composable Kernel headers (from [ROCm/aiter](https://github.com/ROCm/aiter) → `3rdparty/composable_kernel/`):
    ```bash
    git clone https://github.com/ROCm/aiter.git
    ```

### Launch Container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size=64g \
  -v $(pwd)/FastVideo-amd:/workspace/FastVideo \
  -v $(pwd)/aiter/3rdparty/composable_kernel:/workspace/ck_local \
  -w /workspace/FastVideo \
  rocm/primus:v26.2 bash
```

### Build

```bash
cd fastvideo-kernel/csrc/attention/ck_sparse

export CK_DIR=/workspace/ck_local
export GPU_ARCH=gfx950

./build.sh
```

Build takes ~2 minutes. Output: `build/ck_vsa_ops.cpython-3XX-x86_64-linux-gnu.so`.

### Run Tests

```bash
export PYTHONPATH=/workspace/FastVideo/fastvideo-kernel/python

# Full suite: correctness + benchmarks (~5 min)
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --output results.json

# Correctness only (~1 min)
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --quick

# Benchmarks only (~3 min)
python3 fastvideo-kernel/benchmarks/strict_test_ck_fwd.py --bench-only
```

### Python API

```python
import sys, os
sys.path.insert(0, "fastvideo-kernel/csrc/attention/ck_sparse/build")
os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"

import torch
import ck_vsa_ops
from fastvideo_kernel.block_sparse_attn import _map_to_index

# Example: batch=1, heads=12, seq=49152, dim=128, ~10% sparsity
B, H, Sq, D, topk = 1, 12, 49152, 128, 76
BLOCK = 64
nq = nkv = Sq // BLOCK

q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device="cuda")

# Build sparse block map
scores = torch.rand(B, H, nq, nkv, device="cuda")
idx = torch.topk(scores, topk, dim=-1).indices
block_map = torch.zeros(B, H, nq, nkv, dtype=torch.bool, device="cuda")
block_map.scatter_(-1, idx, True)
q2k_index, q2k_num = _map_to_index(block_map)

# Forward
output, lse = ck_vsa_ops.ck_block_sparse_attn_fwd(
    q, k, v, q2k_index, q2k_num, block_m=64
)
# output: [1, 12, 49152, 128] bf16
# lse:    [1, 12, 49152]      fp32
```

---

## 4. Current Scope

| Feature | Status |
|---------|--------|
| Forward pass (bf16) | Supported |
| Forward pass (fp16) | Supported by CK |
| Backward pass | Not yet integrated |
| Symmetric Q/KV lengths | Supported |
| Asymmetric Q/KV lengths | Supported |
| Variable block sizes (< 64 tokens) | Not yet |
| Head dim = 128 | Supported |
| Head dim = 64 | Not yet instantiated |
| Torch autograd integration | Not yet |
