"""
CK-tile block-sparse attention dispatch (forward only, Phase 1).

Loads the compiled CK extension and provides a drop-in replacement for
the Triton forward path used in block_sparse_attn.py.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

_ck_vsa_mod = None


def _load_ck_extension():
    """Lazily load the compiled CK VSA shared library."""
    global _ck_vsa_mod
    if _ck_vsa_mod is not None:
        return _ck_vsa_mod

    # Search paths for the compiled .so
    search_dirs = [
        # Built by csrc/attention/ck_sparse/build.sh
        Path(__file__).resolve().parent.parent.parent / "csrc" / "attention" / "ck_sparse" / "build",
        # Env override
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


def _is_gfx9() -> bool:
    """Check if current GPU is AMD gfx9xx (MI300/MI355 series)."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        # ROCm exposes gcnArchName; fallback to checking device name.
        name = getattr(props, "gcnArchName", "") or props.name
        return "gfx9" in name.lower() or "mi3" in name.lower()
    except Exception:
        return False


def ck_block_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    block_m: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CK-tile block-sparse attention forward.

    Args:
        q: [B, H, Sq, D] bf16/fp16
        k: [B, H, Sk, D]
        v: [B, H, Sk, D]
        q2k_index: [B, H, Q_blocks, max_kv_blks] int32 **absolute** block indices
        q2k_num:   [B, H, Q_blocks] int32 valid block counts
        block_m:   CK tile M dimension (64 or 128)

    Returns:
        (output, lse): output [B,H,Sq,D] same dtype as q; lse [B,H,Sq] fp32
    """
    mod = _load_ck_extension()
    return mod.ck_block_sparse_attn_fwd(q, k, v, q2k_index, q2k_num, block_m)
