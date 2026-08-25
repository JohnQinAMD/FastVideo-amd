"""
CK-tile block-sparse attention dispatch (forward).

Loads the compiled CK extension and provides a drop-in replacement for
the Triton forward path used in block_sparse_attn.py.

Supports:
  - bf16 and fp16
  - head_dim = 64 or 128
  - variable block sizes (partial blocks with < 64 valid tokens)
  - grouped-query and multi-query attention (Hkv dividing Hq)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_ck_vsa_mod = None


def _is_uniform_mask(variable_block_sizes: torch.Tensor) -> bool:
    if variable_block_sizes.numel() == 0:
        return True
    return bool((variable_block_sizes == 64).all().item())


def _load_ck_extension():
    global _ck_vsa_mod
    if _ck_vsa_mod is not None:
        return _ck_vsa_mod

    search_dirs = [
        Path(__file__).resolve().parent.parent.parent / "csrc" / "attention" / "ck_sparse" / "build",
        Path(os.environ.get("CK_VSA_LIB_DIR", "/dev/null")),
    ]

    import importlib, sys as _sys
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("ck_vsa_ops*")):
            if f.suffix not in (".so", ".pyd"):
                continue
            torch.ops.load_library(str(f))
            # Pybind11 module name is the prefix before the first '.' in the
            # filename (the rest is the platform tag inserted by setuptools).
            mod_name = f.name.split(".", 1)[0]
            if str(d) not in _sys.path:
                _sys.path.insert(0, str(d))
            _ck_vsa_mod = importlib.import_module(mod_name)
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
    skip_vbs_correction: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CK-tile block-sparse attention forward.

    Args:
        q: [B, Hq, Sq, D] bf16/fp16, D must be 64 or 128
        k: [B, Hkv, Sk, D] — Hkv must divide Hq (Hkv < Hq is GQA/MQA)
        v: [B, Hkv, Sk, D]
        q2k_index: [B, Hq, Q_blocks, max_kv_blks] int32 absolute block indices,
                   indexed by query head even when Hkv < Hq
        q2k_num:   [B, Hq, Q_blocks] int32 valid block counts
        variable_block_sizes: [num_kv_blocks] int32 — tokens per KV block (1..64)
        block_m:   CK tile M dimension (64)
        skip_vbs_correction: tri-state.
                   None  -> auto-detect: skip iff all variable_block_sizes==64
                   True  -> always skip (caller asserts no partial blocks)
                   False -> always run the correction kernel

    Returns:
        (output, lse): output [B,Hq,Sq,D] same dtype as q; lse [B,Hq,Sq] fp32
    """
    # Auto-detect uniform mask -> skip the vbs correction launch. This reads
    # a device tensor, so pass skip_vbs_correction explicitly to avoid the
    # host sync on a hot path.
    if skip_vbs_correction is None:
        skip_vbs_correction = _is_uniform_mask(variable_block_sizes)

    return _load_ck_extension().ck_block_sparse_attn_fwd(
        q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m,
        skip_vbs_correction
    )