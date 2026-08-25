// Entry points for the VSA forward kernel instantiated with VariableBlockMask.
//
// These live alongside, rather than inside, the CK-generated dispatcher: the
// generated one keys instance selection off traits.mask_type, which describes
// window geometry (causal / local) and has no way to express "mask the tail of
// each KV block". Keeping a separate dispatcher also means running CK codegen
// again does not clobber it.
#pragma once

#include "fmha_fwd_trek.hpp"

namespace fastvideo {

struct vsa_vbs_fwd_args : fmha_vsa_fwd_args
{
    // Valid token count per absolute KV block, int32 [seqlen_k / 64]. A null
    // pointer asserts every block is full, which makes the mask a no-op.
    const void* block_size_ptr = nullptr;
};

template <typename FmhaKernel>
auto vbs_create_kargs_and_grids(const vsa_vbs_fwd_args& args)
{
    // Reuse CK's kargs builder and attach the table afterwards, so the builder
    // itself needs no signature change.
    auto [kargs, grids] =
        fmha_fwd_create_kargs_and_grids<FmhaKernel>(static_cast<const fmha_vsa_fwd_args&>(args));
    kargs.block_size_ptr = args.block_size_ptr;
    return ck_tile::make_tuple(kargs, grids);
}

// Returns the kernel time in ms, or a negative value when no instance matches
// the requested dtype / head dim / block_m.
float fmha_vsa_vbs_fwd(fmha_vsa_fwd_traits t,
                       const vsa_vbs_fwd_args& a,
                       const ck_tile::stream_config& s);

} // namespace fastvideo
