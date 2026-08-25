// SPDX-License-Identifier: MIT
// CK-tile VSA block-sparse attention forward — PyTorch extension wrapper.
//
// Supports:
//   - bf16 and fp16 inputs
//   - head_dim = 64 or 128
//   - variable block sizes (partial blocks with < 64 valid tokens), masked
//     inside the kernel by fastvideo::VariableBlockMask
//   - block_m = 64 tile config

#include <torch/extension.h>
#include <hip/hip_runtime.h>

// CK headers
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"

// Generated CK sparse attention dispatcher + args.
#include "fmha_fwd_trek.hpp"

// Instances bound to fastvideo::VariableBlockMask, for masks with partial blocks.
#include "vsa_vbs_fwd.hpp"

// ---------------------------------------------------------------------------
// Torch-callable forward function.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
ck_block_sparse_attn_fwd(
    torch::Tensor q,                    // [B, Hq, Sq, D]  bf16/fp16
    torch::Tensor k,                    // [B, Hkv, Sk, D]  Hkv may divide Hq (GQA/MQA)
    torch::Tensor v,                    // [B, Hkv, Sk, D]
    torch::Tensor q2k_index,            // [B, Hq, Q_blks, max_kv_blks] int32 absolute
    torch::Tensor q2k_num,              // [B, Hq, Q_blks]              int32 counts
    torch::Tensor variable_block_sizes, // [num_kv_blocks]               int32 (tokens per KV block)
    int block_m,                        // 64 — CK tile selection
    bool uniform_block_sizes = true)    // pass false when any KV block size != 64
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
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");

    const int B  = q.size(0);
    const int H  = q.size(1);
    const int Sq = q.size(2);
    const int D  = q.size(3);
    const int H_kv = k.size(1);
    const int Sk = k.size(2);

    TORCH_CHECK(k.size(0) == B, "k batch (", k.size(0), ") must match q batch (", B, ")");
    TORCH_CHECK(k.size(3) == D, "k head_dim (", k.size(3), ") must match q head_dim (", D, ")");
    // The CK kernel selects a KV head as i_nhead / (nhead_q / nhead_k).
    TORCH_CHECK(H_kv > 0 && H % H_kv == 0,
                "query heads (", H, ") must be divisible by KV heads (", H_kv, ")");
    TORCH_CHECK(q2k_index.size(1) == H && q2k_num.size(1) == H,
                "the block LUT is indexed by query head, so it must have ", H,
                " heads, got ", q2k_index.size(1));
    const int BLOCK_KV = 64;
    // The kernel derives a KV block index from an absolute token index, so the
    // table has to cover every block of the padded sequence.
    TORCH_CHECK(Sk % BLOCK_KV == 0,
                "Sk (", Sk, ") must be a multiple of ", BLOCK_KV);
    TORCH_CHECK(variable_block_sizes.scalar_type() == at::kInt,
                "variable_block_sizes must be int32");
    TORCH_CHECK(variable_block_sizes.numel() == Sk / BLOCK_KV,
                "variable_block_sizes must have one entry per KV block: expected ",
                Sk / BLOCK_KV, ", got ", variable_block_sizes.numel());

    auto o   = torch::empty_like(q);
    auto lse = torch::empty({B, H, Sq}, q.options().dtype(at::kFloat));

    hipStream_t stream = nullptr;

    // --- Build CK args ---
    const float scale_s = 1.0f / std::sqrt(static_cast<float>(D));
    const bool is_bf16 = (q.scalar_type() == at::kBFloat16);

    fmha_vsa_fwd_traits traits;
    traits.hdim_q        = D;
    traits.hdim_v        = D;
    traits.data_type     = is_bf16 ? "bf16" : "fp16";
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;

    fastvideo::vsa_vbs_fwd_args args;
    args.q_ptr               = q.data_ptr();
    args.k_ptr               = k.data_ptr();
    args.v_ptr               = v.data_ptr();
    // The CK pipeline differences the absolute indices inline.
    args.lut_ptr             = q2k_index.data_ptr<int32_t>();
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
    args.nhead_k  = H_kv;
    args.scale_s  = scale_s;

    args.stride_q = D;  args.stride_k = D;  args.stride_v = D;  args.stride_o = D;
    args.nhead_stride_q = Sq * D;  args.nhead_stride_k = Sk * D;
    args.nhead_stride_v = Sk * D;  args.nhead_stride_o = Sq * D;
    args.nhead_stride_lse = Sq;
    args.batch_stride_q = H * Sq * D;  args.batch_stride_k = H_kv * Sk * D;
    args.batch_stride_v = H_kv * Sk * D;  args.batch_stride_o = H * Sq * D;
    args.batch_stride_lse = H * Sq;

    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = 0;
    args.block_m           = block_m;
    args.block_size_ptr    = variable_block_sizes.data_ptr<int32_t>();

    // Partial blocks are masked inside the kernel, before the softmax sees
    // them. Doing it afterwards cannot work: a padded position holds zero K, so
    // it scores exactly 0, and any row whose real scores are all negative ends
    // up with a denominator that is mostly padding. Backing that out means
    // subtracting two nearly equal fp32 numbers and the surviving digits are
    // noise, which showed up end-to-end as output scale errors over 100%.
    //
    // When the caller knows every block is full there is nothing to mask, so
    // take the unmasked instance and skip the per-element predicate entirely.
    ck_tile::stream_config sc{stream, false, 0, 0, 1, false};
    const float ret = uniform_block_sizes
                          ? fmha_vsa_fwd(traits, args, sc)
                          : fastvideo::fmha_vsa_vbs_fwd(traits, args, sc);
    TORCH_CHECK(ret >= 0, "CK fmha_vsa_fwd dispatch failed (no matching kernel). "
                          "hdim=" + std::to_string(D) + " dtype=" + traits.data_type +
                          " block_m=" + std::to_string(block_m) +
                          " masked=" + (uniform_block_sizes ? "0" : "1"));

    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ck_block_sparse_attn_fwd", &ck_block_sparse_attn_fwd,
          "CK-tile block-sparse attention forward",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("q2k_index"), py::arg("q2k_num"),
          py::arg("variable_block_sizes"),
          py::arg("block_m") = 64,
          py::arg("uniform_block_sizes") = true);
}
