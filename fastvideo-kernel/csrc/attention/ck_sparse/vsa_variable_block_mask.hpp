// Variable-block-size mask for the CK VSA forward kernel.
//
// VSA groups the KV sequence into fixed BlockKV-token cubes. A sequence rarely
// fills its last cube of every spatial tile, so the tail rows are padding backed
// by zero-filled K/V. This mask removes them from the softmax entirely.
//
// The alternative - score them and rescale afterwards - is not salvageable in
// fp32. A zero K row scores exactly 0, so it contributes exp2(0 - m) to the
// denominator. Whenever every real score in a row is negative the row max m is
// 0, every padded slot contributes 1, and the denominator degenerates to
// (real_sum + pad_count) with pad_count dominating. Recovering real_sum then
// means evaluating 1 - pad_count/denominator, a subtraction of two nearly equal
// numbers: with 1392 padded slots against 5200 real tokens the true remainder
// sits at 2.9e-7, below one ulp at magnitude 1, and the recovered output scale
// comes back wrong by tens of percent.
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "kernel/fmha_fwd_vsa_kernel.hpp"

namespace fastvideo {

namespace detail {
template <ck_tile::index_t N>
constexpr ck_tile::index_t log2_exact()
{
    static_assert(N > 0 && (N & (N - 1)) == 0, "block size must be a power of two");
    ck_tile::index_t shift = 0;
    for(ck_tile::index_t v = N; v > 1; v >>= 1)
        ++shift;
    return shift;
}
} // namespace detail

// `valid_ptr` holds, per absolute KV block, how many of its BlockKV rows carry
// real tokens; the remainder is padding. A null table means every block is full,
// which makes the mask a no-op and keeps a single instance usable either way.
//
// The VSA pipeline only ever asks a mask for IsEdgeTile and IsOutOfBound - it
// takes its KV loop bounds from the block LUT, not from the mask - so there is
// no tile-range interface to implement here.
template <ck_tile::index_t BlockKV = 64>
struct VariableBlockMask
{
    static constexpr bool IsMasking   = true;
    static constexpr const char* name = "vbsmask";

    static constexpr ck_tile::index_t kShift = detail::log2_exact<BlockKV>();
    static constexpr ck_tile::index_t kMask  = BlockKV - 1;

    CK_TILE_HOST_DEVICE VariableBlockMask(const ck_tile::index_t* valid_ptr_,
                                          ck_tile::index_t y_total_,
                                          ck_tile::index_t x_total_)
        : valid_ptr(valid_ptr_), y_total(y_total_), x_total(x_total_)
    {
    }

    // Valid row count of the block containing absolute KV index x.
    CK_TILE_HOST_DEVICE ck_tile::index_t ValidRows(ck_tile::index_t x) const
    {
        return valid_ptr == nullptr ? BlockKV : valid_ptr[x >> kShift];
    }

    // A K tile is exactly one VSA block wide, so a tile needs per-element
    // checks only when its own block is short. Full blocks - the common case -
    // skip the masking sweep entirely.
    template <ck_tile::index_t TileHeight, ck_tile::index_t TileWidth>
    CK_TILE_HOST_DEVICE bool IsEdgeTile(ck_tile::index_t /*i_y*/,
                                        ck_tile::index_t i_x,
                                        ck_tile::number<TileHeight>,
                                        ck_tile::number<TileWidth>) const
    {
        static_assert(TileWidth == BlockKV,
                      "a K tile must cover exactly one variable-size block");
        return ValidRows(i_x) < BlockKV;
    }

    CK_TILE_HOST_DEVICE bool IsOutOfBound(ck_tile::index_t /*qo_idx*/,
                                          ck_tile::index_t kv_idx) const
    {
        return (kv_idx & kMask) >= ValidRows(kv_idx);
    }

    const ck_tile::index_t* valid_ptr;
    ck_tile::index_t y_total;
    ck_tile::index_t x_total;
};

} // namespace fastvideo

namespace ck_tile {
template <ck_tile::index_t BlockKV>
struct vsa_mask_needs_block_table<fastvideo::VariableBlockMask<BlockKV>> : std::true_type
{
};
} // namespace ck_tile
