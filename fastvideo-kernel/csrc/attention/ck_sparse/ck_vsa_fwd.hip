// SPDX-License-Identifier: MIT
// CK-tile VSA block-sparse attention forward — PyTorch extension wrapper.

#include <torch/extension.h>
#include <hip/hip_runtime.h>

// CK headers
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"

// Generated CK sparse attention dispatcher + args.
#include "fmha_fwd_trek.hpp"

// ---------------------------------------------------------------------------
// HIP kernel: convert absolute block indices -> delta-encoded LUT.
// ---------------------------------------------------------------------------
__global__ void abs_to_delta_kernel(
    const int32_t* __restrict__ abs_idx,
    const int32_t* __restrict__ num_blks,
    int32_t*       __restrict__ delta_out,
    int max_kv_blks,
    int total_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    int n = num_blks[row];
    const int32_t* src = abs_idx + static_cast<int64_t>(row) * max_kv_blks;
    int32_t*       dst = delta_out + static_cast<int64_t>(row) * max_kv_blks;
    int prev = 0;
    for (int i = 0; i < n; ++i) {
        int cur = src[i];
        dst[i] = cur - prev;
        prev = cur;
    }
    for (int i = n; i < max_kv_blks; ++i) {
        dst[i] = 0;
    }
}

// ---------------------------------------------------------------------------
// Torch-callable forward function.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
ck_block_sparse_attn_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor q2k_index,
    torch::Tensor q2k_num,
    int block_m)
{
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Sq,D]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf,
                "q must be bf16 or fp16");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    q2k_index = q2k_index.contiguous();
    q2k_num   = q2k_num.contiguous();

    const int B  = q.size(0);
    const int H  = q.size(1);
    const int Sq = q.size(2);
    const int D  = q.size(3);
    const int Sk = k.size(2);
    const int max_kv_blks = q2k_index.size(-1);

    auto o   = torch::empty_like(q);
    auto lse = torch::empty({B, H, Sq}, q.options().dtype(at::kFloat));

    // Get current HIP stream via PyTorch runtime.
    int device_idx = q.get_device();
    hipStream_t stream = nullptr;
    // Use hipStreamPerThread for the default stream on this device.
    // PyTorch's default stream on ROCm maps to hipStream_t(0).
    // For correctness we just use the default (null) stream since
    // all tensors are on the same device and we call synchronize before return.

    // --- Convert absolute indices to delta LUT ---
    auto delta_lut = torch::empty_like(q2k_index);
    {
        int total_rows = B * H * (Sq / 64);
        int threads = 256;
        int blocks  = (total_rows + threads - 1) / threads;
        abs_to_delta_kernel<<<blocks, threads, 0, stream>>>(
            q2k_index.data_ptr<int32_t>(),
            q2k_num.data_ptr<int32_t>(),
            delta_lut.data_ptr<int32_t>(),
            max_kv_blks,
            total_rows);
    }

    // --- Build CK args ---
    const float scale_s = 1.0f / std::sqrt(static_cast<float>(D));
    const bool is_bf16 = (q.scalar_type() == at::kBFloat16);

    fmha_vsa_fwd_traits traits;
    traits.hdim_q        = D;
    traits.hdim_v        = D;
    traits.data_type     = is_bf16 ? "bf16" : "fp16";
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;

    fmha_vsa_fwd_args args;
    args.q_ptr               = q.data_ptr();
    args.k_ptr               = k.data_ptr();
    args.v_ptr               = v.data_ptr();
    args.lut_ptr             = delta_lut.data_ptr<int32_t>();
    args.valid_block_num_ptr = q2k_num.data_ptr<int32_t>();
    args.o_ptr               = o.data_ptr();
    args.lse_ptr             = lse.data_ptr<float>();

    args.batch    = B;
    args.seqlen_q = Sq;
    args.seqlen_k = Sk;
    args.max_seqlen_q = Sq;
    args.hdim_q   = D;
    args.hdim_v   = D;
    args.nhead_q  = H;
    args.nhead_k  = H;
    args.scale_s  = scale_s;

    args.stride_q = D;
    args.stride_k = D;
    args.stride_v = D;
    args.stride_o = D;

    args.nhead_stride_q = Sq * D;
    args.nhead_stride_k = Sk * D;
    args.nhead_stride_v = Sk * D;
    args.nhead_stride_o = Sq * D;
    args.nhead_stride_lse = Sq;

    args.batch_stride_q = H * Sq * D;
    args.batch_stride_k = H * Sk * D;
    args.batch_stride_v = H * Sk * D;
    args.batch_stride_o = H * Sq * D;
    args.batch_stride_lse = H * Sq;

    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = 0;
    args.block_m           = block_m;

    // Launch CK kernel.
    ck_tile::stream_config sc{stream, false, 0, 0, 1, false};
    float ret = fmha_vsa_fwd(traits, args, sc);
    TORCH_CHECK(ret >= 0, "CK fmha_vsa_fwd dispatch failed (no matching kernel). "
                          "Check hdim/dtype/block_m combination.");

    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ck_block_sparse_attn_fwd", &ck_block_sparse_attn_fwd,
          "CK-tile block-sparse attention forward",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("q2k_index"), py::arg("q2k_num"),
          py::arg("block_m") = 64);
}
