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
# Density above which the HD kernel (bk0=64) outperforms the stock kernel.
HD_DENSITY_THRESHOLD = 0.30

_DELTA_CACHE: dict = {}
_DELTA_CACHE_MAX_ENTRIES = 32


def _delta_cache_key(q2k_index: torch.Tensor, q2k_num: torch.Tensor):
    return (
        q2k_index.data_ptr(), q2k_index._version,
        q2k_num.data_ptr(), q2k_num._version,
        tuple(q2k_index.shape),
    )


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
    q2k_delta: Optional[torch.Tensor] = None,
    skip_vbs_correction: Optional[bool] = None,
    cache_delta: bool = False,
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
        skip_vbs_correction: tri-state.
                   None  -> auto-detect: skip iff all variable_block_sizes==64
                   True  -> always skip (caller asserts no partial blocks)
                   False -> always run the correction kernel
        cache_delta: when True, precomputed delta LUTs are cached across
                   calls keyed on (storage_ptr, version_counter, shape) of
                   q2k_index/q2k_num. Default is False because the
                   torch-side delta computation in this wrapper currently
                   produces output that diverges from the in-kernel
                   abs_to_delta_kernel (cos_sim ≈ 0.94 vs 1.0 expected).
                   Enable only after the torch-side delta has been
                   verified bit-equivalent.

    Returns:
        (output, lse): output [B,H,Sq,D] same dtype as q; lse [B,H,Sq] fp32
    """
    # 0) Auto-detect uniform mask once (used for both vbs-skip and HD dispatch).
    uniform = _is_uniform_mask(variable_block_sizes)

    # 0.5) Density-based kernel selection: HD kernel wins on uniform masks at
    #      >= ~30% density (wider QK K-chunk amortises better). Skip when
    #      block_m != 128 (HD kernel is only emitted for that path) or HD
    #      build is absent.
    mod = None
    # HD .so wins at >=30% density for both block_m=128 (bk0=64 tile) and
    # block_m=64 (the stock 64-tile is identical between the two builds; HD
    # is just preferable because of its tighter call-to-call determinism).
    if uniform and block_m in (64, 128):
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

    # 2) Replay cached delta LUT when the mask hasn't changed.
    if q2k_delta is None and cache_delta:
        key = _delta_cache_key(q2k_index, q2k_num)
        cached = _DELTA_CACHE.get(key)
        if cached is not None:
            q2k_delta = cached
        else:
            # Let CK build it once, then cache. We can't read CK's internal
            # delta back, so we replicate the abs->delta logic here cheaply
            # (one Python launch of a Triton-style scan would be fastest;
            # for simplicity we use a single-shot torch op which is
            # ~tens of microseconds at typical mask shapes).
            with torch.no_grad():
                # Equivalent to: delta[..., 0]   = abs[..., 0]
                #                delta[..., i>0] = abs[..., i] - abs[..., i-1]
                # Out-of-range tail (>= q2k_num) gets zeroed.
                shifted = torch.empty_like(q2k_index)
                shifted[..., 0] = 0
                shifted[..., 1:] = q2k_index[..., :-1]
                delta = q2k_index - shifted
                # Mask the tail.
                ar = torch.arange(q2k_index.shape[-1], device=q2k_index.device,
                                  dtype=torch.int32)
                valid = ar.view(*([1] * (q2k_index.dim() - 1)), -1) < q2k_num.unsqueeze(-1)
                delta = torch.where(valid, delta, torch.zeros_like(delta))
            if len(_DELTA_CACHE) >= _DELTA_CACHE_MAX_ENTRIES:
                _DELTA_CACHE.pop(next(iter(_DELTA_CACHE)))
            _DELTA_CACHE[key] = delta
            q2k_delta = delta

    return mod.ck_block_sparse_attn_fwd(
        q, k, v, q2k_index, q2k_num, variable_block_sizes, block_m,
        q2k_delta, skip_vbs_correction
    )


def clear_delta_cache():
    """Drop all cached delta LUTs. Call when changing models or after a
    significant memory pressure event."""
    _DELTA_CACHE.clear()
