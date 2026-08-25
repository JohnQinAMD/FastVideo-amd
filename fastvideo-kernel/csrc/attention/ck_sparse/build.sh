#!/bin/bash
# Build the CK-tile VSA block-sparse attention PyTorch extension.
#
# Prerequisites:
#   - ROCm with hipcc
#   - PyTorch with ROCm support
#   - CK headers (from aiter-amd or rocm install)
#
# Usage:
#   ./build.sh                          # auto-detect CK path
#   CK_DIR=/path/to/ck ./build.sh      # explicit CK path
#   GPU_ARCH=gfx950 ./build.sh         # explicit GPU arch
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

# --- Locate CK headers ---
if [ -z "${CK_DIR}" ]; then
    # Try common locations.
    for candidate in \
        /opt/rocm/include/.. \
        ; do
        if [ -d "${candidate}/include/ck_tile" ]; then
            CK_DIR="${candidate}"
            break
        fi
    done
fi
if [ -z "${CK_DIR}" ] || [ ! -d "${CK_DIR}/include/ck_tile" ]; then
    echo "ERROR: Cannot find CK headers. Set CK_DIR=/path/to/composable_kernel" >&2
    exit 1
fi
echo "CK_DIR=${CK_DIR}"

# CK sparse attn example dir (contains fmha_fwd_trek.hpp, codegen/, mask.hpp ref).
CK_SPARSE_DIR="${CK_DIR}/example/ck_tile/50_sparse_attn"
CK_FMHA_DIR="${CK_DIR}/example/ck_tile/01_fmha"
CK_EXAMPLE_DIR="${CK_DIR}/example/ck_tile"

# --- GPU arch ---
if [ -z "${GPU_ARCH}" ]; then
    GPU_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep gfx | head -1 || echo "gfx950")
fi
echo "GPU_ARCH=${GPU_ARCH}"

# --- Step 1: Run CK codegen to generate kernel instances ---
CODEGEN_DIR="${BUILD_DIR}/codegen"
VBS_DIR="${BUILD_DIR}/vbs"
mkdir -p "${CODEGEN_DIR}"

if [ ! -f "${CODEGEN_DIR}/fmha_vsa_fwd_api.cpp" ]; then
    echo "Patching CK codegen for head_dim=64 support..."
    python3 "${SCRIPT_DIR}/patch_codegen_d64.py" "${CK_DIR}"

    echo "Running CK codegen..."
    python3 "${CK_SPARSE_DIR}/generate.py" \
        --api fwd_vsa --receipt 600 \
        --output_dir "${CODEGEN_DIR}"
    echo "Codegen produced $(ls "${CODEGEN_DIR}"/*.cpp | wc -l) files."
else
    echo "Codegen already done (${CODEGEN_DIR}/fmha_vsa_fwd_api.cpp exists)."
fi

# Let the variable-block mask reach its valid-length table. Runs unconditionally
# because it edits a CK header the codegen step above does not touch, and a
# freshly checked out CK tree needs it even when codegen output is cached. It
# rewrites a header every instance includes, so drop stale objects when it fires
# (the object rules below only compare source timestamps).
VSA_KERNEL_HDR="${CK_DIR}/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp"
if ! grep -q "vsa_mask_needs_block_table" "${VSA_KERNEL_HDR}"; then
    python3 "${SCRIPT_DIR}/patch_ck_vbs_mask.py" "${CK_DIR}"
    rm -rf "${BUILD_DIR}/obj"
else
    echo "CK variable-block mask hook already present."
fi

echo "Generating variable-block-mask instances..."
python3 "${SCRIPT_DIR}/gen_vbs_instances.py" "${VBS_DIR}"

# --- Step 2: Compile all .cpp/.hip sources into .o files ---
HIPCC="${ROCM_PATH:-/opt/rocm}/bin/hipcc"
if [ ! -x "${HIPCC}" ]; then
    HIPCC="hipcc"
fi

TORCH_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)
TORCH_INC=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")
TORCH_LIB=$(python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L'+p for p in library_paths()]))")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

COMMON_FLAGS=(
    -std=c++17 -fPIC -O3
    --offload-arch=${GPU_ARCH}
    -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
    -DCK_TILE_FMHA_FWD_FAST_EXP2
    ${CK_VSA_EXTRA_FLAGS:-}
    -Wno-undefined-func-template
    -Wno-float-equal
    -I"${CK_DIR}/include"
    -I"${CK_DIR}/include/ck_tile/ops/sparse_attn"
    -I"${SCRIPT_DIR}"
    -I"${CK_SPARSE_DIR}"
    -I"${CK_FMHA_DIR}"
    -I"${CK_EXAMPLE_DIR}"
    -I"${PYTHON_INC}"
    ${TORCH_INC}
)

OBJ_DIR="${BUILD_DIR}/obj"
mkdir -p "${OBJ_DIR}"

echo "Compiling kernel instances..."
OBJS=()

# Compile CK codegen'd kernel files and the variable-block-mask instances (in
# parallel).
for src in "${CODEGEN_DIR}"/*.cpp "${VBS_DIR}"/*.cpp; do
    base=$(basename "${src}" .cpp)
    obj="${OBJ_DIR}/${base}.o"
    OBJS+=("${obj}")
    if [ ! -f "${obj}" ] || [ "${src}" -nt "${obj}" ]; then
        echo "  hipcc ${base}.cpp"
        ${HIPCC} "${COMMON_FLAGS[@]}" -c "${src}" -o "${obj}" &
    fi
done

# Limit parallel jobs.
wait

# Compile the PyTorch wrapper.
echo "Compiling PyTorch wrapper..."
WRAPPER_OBJ="${OBJ_DIR}/ck_vsa_fwd.o"
OBJS+=("${WRAPPER_OBJ}")
${HIPCC} "${COMMON_FLAGS[@]}" \
    -DTORCH_EXTENSION_NAME=ck_vsa_ops \
    -c "${SCRIPT_DIR}/ck_vsa_fwd.hip" \
    -o "${WRAPPER_OBJ}"

# --- Step 3: Link into shared library ---
SO_OUT="${BUILD_DIR}/ck_vsa_ops$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
echo "Linking ${SO_OUT}..."
${HIPCC} -shared -o "${SO_OUT}" "${OBJS[@]}" \
    ${TORCH_LIB} -ltorch -ltorch_hip -lc10 -lc10_hip -ltorch_python

echo ""
echo "Build complete: ${SO_OUT}"
echo "To use: import torch; torch.ops.load_library('${SO_OUT}')"
