#!/usr/bin/env python3
"""
Patch the CK fmha_fwd_vsa.py codegen to uncomment hdim=(64,64) tile sizes.
Run this before generate.py to enable head_dim=64 kernel instances.

Usage:
    python3 patch_codegen_d64.py /path/to/composable_kernel
"""
import sys, re
from pathlib import Path

def patch(ck_dir: str):
    f = Path(ck_dir) / "example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_vsa.py"
    text = f.read_text()

    # Uncomment the (64, 64) entry in get_hdim_tile_size_dict.
    # Original has lines like:
    #   # (64, 64)  : [FmhaFwdTileSize(128, 64,  32, 64,  32,  64, ...)]
    # We uncomment the last tile (128x64 with 32x32x16 warp) which matches our kM0=128,kN0=64 pattern.
    # Also add a 64x64 tile for kM0=64.

    # Find the block between "(128, 128):" and the closing bracket
    # Insert the (64,64) entry right before (128,128)

    new_entry = '''                (64, 64)  : [
                    FmhaFwdTileSize(  # kM0=64, kN0=64 — matches VSA 64-token blocks
                        128, 64, 32, 64, 32, 64,
                        4, 1, 1,  4, 1, 1,
                        32, 32, 16,  32, 32, 16,  -1),
                    FmhaFwdTileSize(  # kM0=64, kN0=64 — smaller warp tile
                        64, 64, 32, 64, 32, 64,
                        2, 1, 1,  2, 1, 1,
                        32, 32, 16,  32, 32, 16,  -1),
                ],
'''

    # Insert before the (128, 128) line
    marker = "                (128, 128): ["
    if "(64, 64)" not in text.split("get_hdim_tile_size_dict")[1].split("get_hdim_tile_size_dict")[0]:
        text = text.replace(marker, new_entry + marker)
        f.write_text(text)
        print(f"Patched {f}: added (64,64) hdim tile sizes")
    else:
        print(f"Already patched: {f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 patch_codegen_d64.py /path/to/composable_kernel")
        sys.exit(1)
    patch(sys.argv[1])
