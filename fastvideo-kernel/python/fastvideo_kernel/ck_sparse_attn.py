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
_ck_vsa_hd_mod = None

# Density above which the HD build is used instead of the stock one. Infinite
# (i.e. HD never selected) by default: the bk0=64 tile the HD build was created
# to introduce now ships in the stock CK codegen, so both builds run the same
# tile, and the HD pipeline's remaining difference — a second QK accumulator —
# raises VGPR use from 96 to 104. On gfx950 that crosses an occupancy boundary
# (512 registers/SIMD: 5 waves at 96, 4 at 104) and costs ~4% throughput.
# Set FASTVIDEO_KERNEL_CK_HD_THRESHOLD to re-enable on other architectures.
HD_DENSITY_THRESHOLD = float(os.environ.get("FASTVIDEO_KERNEL_CK_HD_THRESHOLD", "inf"))


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


def _load_ck_hd_extension():
    """Return the high-density CK extension (bk0=64), or None if not built."""
    global _ck_vsa_hd_mod
    if _ck_vsa_hd_mod is not None:
        return _ck_vsa_hd_mod
    import importlib, sys as _sys
    search_dirs = [
        Path(__file__).resolve().parent.parent.parent / "csrc" / "attention" / "ck_sparse" / "build_hd",
        Path(os.environ.get("CK_VSA_HD_LIB_DIR", "/dev/null")),
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("ck_vsa_ops_hd*")):
            if f.suffix not in (".so", ".pyd"):
                continue
            torch.ops.load_library(str(f))
            mod_name = f.name.split(".", 1)[0]
            if str(d) not in _sys.path:
                _sys.path.insert(0, str(d))
            _ck_vsa_hd_mod = importlib.import_module(mod_name)
            return _ck_vsa_hd_mod
    return None


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
        q: [B, H, Sq, D] bf16/fp16, D must be 64 or 128
        k: [B, H, Sk, D]
        v: [B, H, Sk, D]
        q2k_index: [B, H, Q_blocks, max_kv_blks] int32 absolute block indices
        q2k_num:   [B, H, Q_blocks] int32 valid block counts
        variable_block_sizes: [num_kv_blocks] int32 — tokens per KV block (1..64)
        block_m:   CK tile M dimension (64)
        skip_vbs_correction: tri-state.
                   None  -> auto-detect: skip iff all variable_block_sizes==64
                   True  -> always skip (caller asserts no partial blocks)
                   False -> always run the correction kernel

    Returns:
        (output, lse): output [B,H,Sq,D] same dtype as q; lse [B,H,Sq] fp32
    """
    # 0) Auto-detect uniform mask once (used for both vbs-skip and HD dispatch).
    uniform = _is_uniform_mask(variable_block_sizes)

    # 0.5) Optional density-based kernel selection. Disabled unless
    #      HD_DENSITY_THRESHOLD is finite — see the note on that constant.
    mod = None
    if uniform and block_m in (64, 128) and HD_DENSITY_THRESHOLD != float("inf"):
        if q2k_num.numel() > 0:
            top1 = int(q2k_num.flatten()[0].item())
            density = top1 / max(1, q2k_index.shape[-1])
            if density >= HD_DENSITY_THRESHOLD:
                hd = _load_ck_hd_extension()
                if hd is not None:
                    mod = hd
    if mod is None:
        mod = _load_ck_extension()

    # 1) Auto-detect uniform mask -> skip vbs correction launch.
    if skip_vbs_correction is None:
        skip_vbs_correction = uniform

    return mod.ck_block_sparse_attn_fwd(
        q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m,
        skip_vbs_correction
    )