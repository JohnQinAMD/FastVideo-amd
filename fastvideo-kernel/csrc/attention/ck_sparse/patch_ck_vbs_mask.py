#!/usr/bin/env python3
"""
Teach the CK VSA kernel to hand a per-block valid-length table to its mask.

VSA tiles the KV sequence into fixed 64-token cubes, so the last cube of a
sequence is usually part padding backed by zero-filled K/V. Those rows have to
be masked out *before* the softmax: a zero K row scores exactly 0, so whenever
every real score in a row is negative the padding count becomes the softmax
denominator, and no amount of post-hoc correction recovers the real one from
fp32 (see docs/ck_block_sparse_attn_guide.md).

The mask object is built inside the kernel from its kargs, so a mask that needs
the table needs a pointer to reach it. Two edits do that:

  1. `FmhaFwdMaskKargs` gains a `block_size_ptr` member. It is mask-specific
     state and only present when the instance has masking enabled, so this is
     where it belongs. It defaults to nullptr, and `MakeKargs` keeps working
     untouched because the member is aggregate-initialised from `{}`; callers
     that want the table assign it on the returned kargs.
  2. The mask construction site dispatches on an opt-in trait, so masks that
     do not ask for the table are built exactly as before.

This edits the CK checkout in place, like patch_codegen_d64.py. Every anchor is
matched exactly and a miss is fatal rather than silently skipped: a CK bump that
moves this code must fail the build loudly, not quietly go back to producing
contaminated denominators.

Usage:
    python3 patch_ck_vbs_mask.py /path/to/composable_kernel
"""
import sys
from pathlib import Path

KERNEL_REL = "include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp"

SENTINEL = "vsa_mask_needs_block_table"

TRAIT_ANCHOR = """namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdVSAKernel
{"""

TRAIT_PATCHED = """namespace ck_tile {

// Opt-in hook for masks that are driven by a per-KV-block valid-length table
// rather than by a window geometry. Specialised by the mask's own header; the
// primary template keeps every existing mask on the original construction path.
template <typename Mask>
struct vsa_mask_needs_block_table : std::false_type
{
};

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdVSAKernel
{"""

KARGS_ANCHOR = """    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };"""

KARGS_PATCHED = """    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
        // Valid token count per absolute KV block, int32 [seqlen_k / kN0].
        // Only read by masks that opt into vsa_mask_needs_block_table.
        const void* block_size_ptr = nullptr;
    };"""

MASK_ANCHOR = """        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();"""

MASK_PATCHED = """        FmhaMask mask = [&]() {
            if constexpr(vsa_mask_needs_block_table<FmhaMask>::value)
                return FmhaMask{reinterpret_cast<const ck_tile::index_t*>(kargs.block_size_ptr),
                                kargs.seqlen_q,
                                kargs.seqlen_k};
            else if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();"""

EDITS = (
    ("trait hook", TRAIT_ANCHOR, TRAIT_PATCHED),
    ("mask kargs", KARGS_ANCHOR, KARGS_PATCHED),
    ("mask construction", MASK_ANCHOR, MASK_PATCHED),
)


def patch(ck_dir: str) -> None:
    path = Path(ck_dir) / KERNEL_REL
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} not found; is CK_DIR={ck_dir} a composable_kernel checkout?")

    text = path.read_text()
    if SENTINEL in text:
        print(f"Already patched: {path}")
        return

    for name, anchor, patched in EDITS:
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(
                f"ERROR: the '{name}' anchor matched {count} times in {path}, expected exactly 1.\n"
                "CK has moved this code, so the variable-block mask can no longer reach its\n"
                "valid-length table. Re-derive the anchor before building: without it the\n"
                "kernel silently scores zero-filled padding and corrupts the softmax denominator."
            )
        text = text.replace(anchor, patched)

    path.write_text(text)
    print(f"Patched {path}: variable-block mask can now read its valid-length table")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: patch_ck_vbs_mask.py /path/to/composable_kernel")
    patch(sys.argv[1])
