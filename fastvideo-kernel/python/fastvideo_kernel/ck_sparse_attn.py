"""
CK-tile block-sparse attention dispatch (forward).

Loads the compiled CK extension and provides a drop-in replacement for
the Triton forward path used in block_sparse_attn.py.

Supports:
  - bf16 and fp16
  - head_dim = 64 or 128
  - variable block sizes (partial blocks with < 64 valid tokens)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_ck_vsa_mod = None


def _load_ck_extension():
    global _ck_vsa_mod
    if _ck_vsa_mod is not None:
        return _ck_vsa_mod

    search_dirs = [
        Path(__file__).resolve().parent.parent.parent / "csrc" / "attention" / "ck_sparse" / "build",
        Path(os.environ.get("CK_VSA_LIB_DIR", "/dev/null")),
    ]

    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in d.glob("ck_vsa_ops*"):
            if f.suffix in (".so", ".pyd"):
                torch.ops.load_library(str(f))
                import importlib
                _ck_vsa_mod = importlib.import_module(f.stem)
                return _ck_vsa_mod

    raise ImportError(
        "CK VSA extension not found. Build it with:\n"
        "  cd fastvideo-kernel/csrc/attention/ck_sparse && ./build.sh"
    )


def ck_block_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_m: int = 64,
    q2k_delta: Optional[torch.Tensor] = None,
    skip_vbs_correction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CK-tile block-sparse attention forward.

    Args:
        q: [B, H, Sq, D] bf16/fp16, D must be 64 or 128
        k: [B, H, Sk, D]
        v: [B, H, Sk, D]
        q2k_index: [B, H, Q_blocks, max_kv_blks] int32 absolute block indices
        q2k_num:   [B, H, Q_blocks] int32 valid block counts
        variable_block_sizes: [num_kv_blocks] int32 — tokens per KV block (1..64)
        block_m:   CK tile M dimension (64)
        q2k_delta: [B, H, Q_blocks, max_kv_blks] int32 delta-encoded LUT.
                   When provided (e.g. from map_to_index_and_delta), the
                   HIP abs→delta kernel is skipped, saving one launch.
        skip_vbs_correction: When True, skip the VBS output correction
                   kernel entirely. Safe when all block sizes == 64.

    Returns:
        (output, lse): output [B,H,Sq,D] same dtype as q; lse [B,H,Sq] fp32
    """
    mod = _load_ck_extension()
    return mod.ck_block_sparse_attn_fwd(
        q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m,
        q2k_delta, skip_vbs_correction
    )
