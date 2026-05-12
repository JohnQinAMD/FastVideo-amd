// SPDX-License-Identifier: MIT
// CK-tile VSA block-sparse attention forward — PyTorch extension wrapper.
//
// Supports:
//   - bf16 and fp16 inputs
//   - head_dim = 64 or 128
//   - variable block sizes (partial blocks with < 64 valid tokens)
//   - block_m = 64 tile config

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
// HIP kernel: variable block size LSE correction.
//
// When variable_block_sizes[kv_blk] < BLOCK (64), the padded K/V positions
// are zeros. CK treats them as real tokens: QK=0 -> exp2(0)=1 in softmax
// denominator. This dilutes attention weights.
//
// Correction: for each Q token i,
//   P_i = total padded positions across attended KV blocks
//   output_correct_i = output_ck_i * exp2(lse_ck_i) / (exp2(lse_ck_i) - P_i)
//   lse_correct_i    = log2(exp2(lse_ck_i) - P_i)
//
// Grid: one thread per Q token (B * H * Sq).
// ---------------------------------------------------------------------------
__global__ void vbs_lse_correction_kernel(
    float*         __restrict__ o_ptr,       // [B*H*Sq*D] — output (in-place, as float for correction)
    float*         __restrict__ lse_ptr,     // [B*H*Sq]
    const int32_t* __restrict__ q2k_idx,     // [B*H*Q_blks, max_kv_blks] absolute indices
    const int32_t* __restrict__ q2k_num,     // [B*H*Q_blks]
    const int32_t* __restrict__ vbs,         // [num_kv_blocks]
    int Q_blks,
    int max_kv_blks,
    int block_q,
    int block_kv,
    int D,
    int total_q_tokens)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_q_tokens) return;

    int q_blk = tid / block_q;
    int n = q2k_num[q_blk];
    const int32_t* idx_row = q2k_idx + static_cast<int64_t>(q_blk) * max_kv_blks;

    // Count total padded positions for this Q token.
    int pad_count = 0;
    for (int i = 0; i < n; ++i) {
        int kv_blk = idx_row[i];
        int bs = vbs[kv_blk];
        pad_count += (block_kv - bs);
    }

    if (pad_count == 0) return;  // all blocks full, no correction needed

    // Apply correction.
    float lse_val = lse_ptr[tid];
    float denom_ck = exp2f(lse_val);
    float denom_correct = denom_ck - static_cast<float>(pad_count);

    // Guard: if denom_correct <= 0, all valid positions had -inf scores.
    if (denom_correct <= 0.0f) {
        lse_ptr[tid] = -1e30f;
        // Zero out the output for this token.
        float* o_row = o_ptr + static_cast<int64_t>(tid) * D;
        for (int d = 0; d < D; ++d) o_row[d] = 0.0f;
        return;
    }

    float scale = denom_ck / denom_correct;
    lse_ptr[tid] = log2f(denom_correct);

    // Scale output in-place.
    float* o_row = o_ptr + static_cast<int64_t>(tid) * D;
    for (int d = 0; d < D; ++d) {
        o_row[d] *= scale;
    }
}

// bf16/fp16 wrapper for in-place output correction.
template <typename T>
__global__ void vbs_output_correction_kernel(
    T*             __restrict__ o_ptr,       // [total_q_tokens, D]
    const float*   __restrict__ lse_ptr,     // [total_q_tokens] — already corrected
    const float*   __restrict__ lse_ck_ptr,  // [total_q_tokens] — original CK LSE
    const int32_t* __restrict__ q2k_idx,
    const int32_t* __restrict__ q2k_num,
    const int32_t* __restrict__ vbs,
    int max_kv_blks,
    int block_q,
    int block_kv,
    int D,
    int total_q_tokens)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_q_tokens) return;

    int q_blk = tid / block_q;
    int n = q2k_num[q_blk];
    const int32_t* idx_row = q2k_idx + static_cast<int64_t>(q_blk) * max_kv_blks;

    int pad_count = 0;
    for (int i = 0; i < n; ++i) {
        pad_count += (block_kv - vbs[idx_row[i]]);
    }
    if (pad_count == 0) return;

    float lse_val = lse_ck_ptr[tid];
    float denom_ck = exp2f(lse_val);
    float denom_correct = denom_ck - static_cast<float>(pad_count);

    if (denom_correct <= 0.0f) {
        T* o_row = o_ptr + static_cast<int64_t>(tid) * D;
        for (int d = 0; d < D; ++d) o_row[d] = static_cast<T>(0.0f);
        return;
    }

    float scale = denom_ck / denom_correct;
    T* o_row = o_ptr + static_cast<int64_t>(tid) * D;
    for (int d = 0; d < D; ++d) {
        o_row[d] = static_cast<T>(static_cast<float>(o_row[d]) * scale);
    }
}

// ---------------------------------------------------------------------------
// Torch-callable forward function.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
ck_block_sparse_attn_fwd(
    torch::Tensor q,                    // [B, H, Sq, D]  bf16/fp16
    torch::Tensor k,                    // [B, H, Sk, D]
    torch::Tensor v,                    // [B, H, Sk, D]
    torch::Tensor q2k_index,            // [B, H, Q_blks, max_kv_blks]  int32 absolute
    torch::Tensor q2k_num,              // [B, H, Q_blks]               int32 counts
    torch::Tensor variable_block_sizes, // [num_kv_blocks]               int32 (tokens per KV block)
    int block_m,                        // 64 — CK tile selection
    c10::optional<torch::Tensor> q2k_delta_opt = c10::nullopt,
                                        // accepted for API back-compat; ignored
                                        // (CK pipeline consumes absolute indices directly).
    bool skip_vbs_correction = true)    // pass false when any KV block size != 64
{
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Sq,D]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf,
                "q must be bf16 or fp16");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q2k_index.is_contiguous(), "q2k_index must be contiguous");
    TORCH_CHECK(q2k_num.is_contiguous(), "q2k_num must be contiguous");
    TORCH_CHECK(variable_block_sizes.is_contiguous(),
                "variable_block_sizes must be contiguous");

    const int B  = q.size(0);
    const int H  = q.size(1);
    const int Sq = q.size(2);
    const int D  = q.size(3);
    const int Sk = k.size(2);
    const int max_kv_blks = q2k_index.size(-1);
    const int BLOCK_KV = 64;
    const int Q_blks = q2k_index.size(2);

    auto o   = torch::empty_like(q);
    auto lse = torch::empty({B, H, Sq}, q.options().dtype(at::kFloat));

    hipStream_t stream = nullptr;

    // CK pipeline computes delta inline from absolute indices; pass q2k_index directly.
    (void)q2k_delta_opt;
    const torch::Tensor& delta_lut = q2k_index;

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

    args.stride_q = D;  args.stride_k = D;  args.stride_v = D;  args.stride_o = D;
    args.nhead_stride_q = Sq * D;  args.nhead_stride_k = Sk * D;
    args.nhead_stride_v = Sk * D;  args.nhead_stride_o = Sq * D;
    args.nhead_stride_lse = Sq;
    args.batch_stride_q = H * Sq * D;  args.batch_stride_k = H * Sk * D;
    args.batch_stride_v = H * Sk * D;  args.batch_stride_o = H * Sq * D;
    args.batch_stride_lse = H * Sq;

    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = 0;
    args.block_m           = block_m;

    // Launch CK kernel.
    ck_tile::stream_config sc{stream, false, 0, 0, 1, false};
    float ret = fmha_vsa_fwd(traits, args, sc);
    TORCH_CHECK(ret >= 0, "CK fmha_vsa_fwd dispatch failed (no matching kernel). "
                          "hdim=" + std::to_string(D) + " dtype=" + traits.data_type +
                          " block_m=" + std::to_string(block_m));

    // --- Variable block size correction ---
    // When all KV blocks are full (block_size == 64), no correction is
    // needed. The caller can signal this via skip_vbs_correction=true to
    // eliminate the kernel launch overhead (~5 µs).
    if (!skip_vbs_correction) {
        int total_q_tokens = B * H * Sq;
        int threads = 256;
        int blocks  = (total_q_tokens + threads - 1) / threads;
        if (is_bf16) {
            vbs_output_correction_kernel<hip_bfloat16><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<hip_bfloat16*>(o.data_ptr()),
                lse.data_ptr<float>(),
                lse.data_ptr<float>(),
                q2k_index.data_ptr<int32_t>(),
                q2k_num.data_ptr<int32_t>(),
                variable_block_sizes.data_ptr<int32_t>(),
                max_kv_blks, block_m, BLOCK_KV, D, total_q_tokens);
        } else {
            vbs_output_correction_kernel<__half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<__half*>(o.data_ptr()),
                lse.data_ptr<float>(),
                lse.data_ptr<float>(),
                q2k_index.data_ptr<int32_t>(),
                q2k_num.data_ptr<int32_t>(),
                variable_block_sizes.data_ptr<int32_t>(),
                max_kv_blks, block_m, BLOCK_KV, D, total_q_tokens);
        }
    }

    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ck_block_sparse_attn_fwd", &ck_block_sparse_attn_fwd,
          "CK-tile block-sparse attention forward",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("q2k_index"), py::arg("q2k_num"),
          py::arg("variable_block_sizes"),
          py::arg("block_m") = 64,
          py::arg("q2k_delta") = c10::nullopt,
          py::arg("skip_vbs_correction") = true);
}
