#!/usr/bin/env python3
"""Patch CK codegen to use the HD bk0=64 K-tile for the d128 path."""
import sys
from pathlib import Path

HD_TILE_TAG = "# patched-hd-bk064"

ORIGINAL_BLOCK = """                    FmhaFwdTileSize(  # fmt: skip
                        128,
                        64,
                        32,
                        128,
                        32,  # kK1=32 (16 triggers LDS descriptor assertion with VSA policy)
                        128,
                        4,
                        1,
                        1,
                        4,
                        1,
                        1,
                        32,
                        32,
                        16,
                        32,
                        32,
                        16,
                        -1,
                    ),"""

ORIGINAL_BLOCK_ALT = """                    FmhaFwdTileSize(  # fmt: skip
                        128,
                        64,
                        32,
                        128,
                        16,
                        128,
                        4,
                        1,
                        1,
                        4,
                        1,
                        1,
                        32,
                        32,
                        16,
                        32,
                        32,
                        16,
                        -1,
                    ),"""

HD_REPLACEMENT = """                    FmhaFwdTileSize(  # patched-hd-bk064
                        128,
                        64,
                        64,
                        128,
                        32,
                        128,
                        4,
                        1,
                        1,
                        4,
                        1,
                        1,
                        32,
                        32,
                        16,
                        32,
                        32,
                        16,
                        -1,
                    ),"""


def patch(ck_dir):
    f = Path(ck_dir) / "example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_vsa.py"
    text = f.read_text()
    if HD_TILE_TAG in text:
        print(f"Already HD-patched: {f}")
        return
    if "kK0=64" in text:
        print(f"HD bk0=64 tile already present in upstream codegen: {f}")
        return
    if ORIGINAL_BLOCK in text:
        text_patched = text.replace(ORIGINAL_BLOCK, HD_REPLACEMENT)
    elif ORIGINAL_BLOCK_ALT in text:
        text_patched = text.replace(ORIGINAL_BLOCK_ALT, HD_REPLACEMENT)
    else:
        raise RuntimeError(f"stock tile not found in {f}")
    f.write_text(text_patched)
    print(f"Patched {f}")


if __name__ == "__main__":
    patch(sys.argv[1])
