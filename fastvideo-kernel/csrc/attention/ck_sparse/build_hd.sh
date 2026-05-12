#!/bin/bash
# Build the HIGH-DENSITY variant of the CK-tile VSA block-sparse attention
# extension. Produces ck_vsa_ops_hd.so in build_hd/ alongside a patched copy
# of the CK source tree at build_hd/_ck_src_patched/. The original build/
# tree (ck_vsa_ops.so) is left untouched.
#
# Usage:
#   ./build_hd.sh                        # auto-detect CK_DIR, gfx950
#   CK_DIR=/path/to/ck ./build_hd.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_hd"
mkdir -p "${BUILD_DIR}"

# --- Locate CK source ---
# Prefer the CK that the running container ships with (it must match the
# fmha_vsa_fwd_args fields the wrapper expects). The aiter-amd repo on
# /mnt/vast may be older/newer and have different field names.
if [ -z "${CK_DIR}" ]; then
    for candidate in \
        /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel \
        /mnt/vast/john/rocm-dynamo/aiter-amd/3rdparty/composable_kernel \
        /opt/rocm/include/.. \
        ; do
        if [ -d "${candidate}/include/ck_tile" ]; then
            CK_DIR="${candidate}"
            break
        fi
    done
fi
if [ -z "${CK_DIR}" ] || [ ! -d "${CK_DIR}/include/ck_tile" ]; then
    echo "ERROR: Cannot find CK source. Set CK_DIR=/path/to/composable_kernel" >&2
    exit 1
fi
echo "CK_DIR=${CK_DIR}"

# --- Mirror CK source into a build-local patched copy ---
PATCHED_CK="${BUILD_DIR}/_ck_src_patched"
SPARSE_REL="example/ck_tile/50_sparse_attn"
if [ ! -d "${PATCHED_CK}/${SPARSE_REL}" ]; then
    echo "Cloning CK ${SPARSE_REL} into ${PATCHED_CK}..."
    mkdir -p "${PATCHED_CK}"
    # We only need the example dir + headers; symlink everything else to save space.
    for sub in include 3rdparty cmake; do
        [ -e "${CK_DIR}/${sub}" ] && ln -sfn "${CK_DIR}/${sub}" "${PATCHED_CK}/${sub}"
    done
    mkdir -p "${PATCHED_CK}/example/ck_tile"
    cp -a "${CK_DIR}/example/ck_tile/01_fmha" "${PATCHED_CK}/example/ck_tile/" 2>/dev/null || true
    cp -a "${CK_DIR}/${SPARSE_REL}" "${PATCHED_CK}/${SPARSE_REL}"
fi

# --- Apply patches: d64 hdim coverage + HD bk0=64 bk1=64 tile ---
python3 "${SCRIPT_DIR}/patch_codegen_d64.py" "${PATCHED_CK}"
python3 "${SCRIPT_DIR}/patch_codegen_hd.py" "${PATCHED_CK}"

# --- GPU arch ---
if [ -z "${GPU_ARCH}" ]; then
    GPU_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep gfx | head -1 || echo "gfx950")
fi
echo "GPU_ARCH=${GPU_ARCH}"

CK_SPARSE_DIR="${PATCHED_CK}/${SPARSE_REL}"
CK_FMHA_DIR="${PATCHED_CK}/example/ck_tile/01_fmha"
CK_EXAMPLE_DIR="${PATCHED_CK}/example/ck_tile"

# --- Step 1: Codegen ---
CODEGEN_DIR="${BUILD_DIR}/codegen"
mkdir -p "${CODEGEN_DIR}"
if [ ! -f "${CODEGEN_DIR}/fmha_vsa_fwd_api.cpp" ]; then
    echo "Running CK codegen (HD variant)..."
    python3 "${CK_SPARSE_DIR}/generate.py" \
        --api fwd_vsa --receipt 600 \
        --output_dir "${CODEGEN_DIR}"
    echo "Codegen produced $(ls "${CODEGEN_DIR}"/*.cpp | wc -l) files."
else
    echo "Codegen already done."
fi

# --- Step 2: Compile ---
HIPCC="${ROCM_PATH:-/opt/rocm}/bin/hipcc"
[ -x "${HIPCC}" ] || HIPCC="hipcc"

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
    -I"${SCRIPT_DIR}/build_hd/include_override/example/ck_tile/50_sparse_attn"
    -I"${SCRIPT_DIR}/build_hd/include_override/ck_tile/ops/sparse_attn"
    -I"${SCRIPT_DIR}/build_hd/include_override"
    -I"${PATCHED_CK}/include"
    -I"${PATCHED_CK}/include/ck_tile/ops/sparse_attn"
    -I"${CK_SPARSE_DIR}"
    -I"${CK_FMHA_DIR}"
    -I"${CK_EXAMPLE_DIR}"
    -I"${PYTHON_INC}"
    ${TORCH_INC}
)

OBJ_DIR="${BUILD_DIR}/obj"
mkdir -p "${OBJ_DIR}"

echo "Compiling kernel instances (HD)..."
OBJS=()
for src in "${CODEGEN_DIR}"/*.cpp; do
    base=$(basename "${src}" .cpp)
    obj="${OBJ_DIR}/${base}.o"
    OBJS+=("${obj}")
    if [ ! -f "${obj}" ] || [ "${src}" -nt "${obj}" ]; then
        echo "  hipcc ${base}.cpp"
        ${HIPCC} "${COMMON_FLAGS[@]}" -c "${src}" -o "${obj}" &
    fi
done
wait

echo "Compiling PyTorch wrapper (module name = ck_vsa_ops_hd)..."
WRAPPER_OBJ="${OBJ_DIR}/ck_vsa_fwd.o"
OBJS+=("${WRAPPER_OBJ}")
${HIPCC} "${COMMON_FLAGS[@]}" \
    -DTORCH_EXTENSION_NAME=ck_vsa_ops_hd \
    -c "${SCRIPT_DIR}/ck_vsa_fwd.hip" \
    -o "${WRAPPER_OBJ}"

# --- Step 3: Link ---
SO_OUT="${BUILD_DIR}/ck_vsa_ops_hd$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
echo "Linking ${SO_OUT}..."
${HIPCC} -shared -o "${SO_OUT}" "${OBJS[@]}" \
    ${TORCH_LIB} -ltorch -ltorch_hip -lc10 -lc10_hip -ltorch_python

echo ""
echo "Build complete: ${SO_OUT}"
