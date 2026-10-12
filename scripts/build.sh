#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: activate a Python 3.12 Conda environment before building." >&2
  exit 1
fi

if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  echo "error: PipeSwift requires Python 3.12." >&2
  exit 1
fi

PIPESWIFT_INSTALL_TOOLCHAIN="${PIPESWIFT_INSTALL_TOOLCHAIN:-1}"
if [[ "${PIPESWIFT_INSTALL_TOOLCHAIN}" == "1" ]]; then
  conda install --yes -c nvidia -c conda-forge \
    cuda-toolkit=12.9 gcc=14.3.0 rdma-core
fi

python -m pip install --upgrade pip setuptools wheel

if ! python -c 'import torch; assert torch.__version__.split("+")[0].startswith("2.9."); assert torch.version.cuda == "12.9"' 2>/dev/null; then
  python -m pip install \
    torch==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu129
fi

python -m pip install \
  anyio \
  build \
  fastapi \
  'httpx[socks]' \
  huggingface_hub \
  ninja \
  pandas \
  pre-commit \
  protobuf \
  psutil \
  pyarrow \
  pydantic \
  pyzmq \
  safetensors \
  sortedcontainers \
  starlette \
  tiktoken \
  transformers \
  uvicorn \
  xxhash

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
export FLASH_ATTN_CUDA_ARCHS=90
export FLASHINFER_CUDA_ARCH_LIST=9.0a
export FLASH_MLA_DISABLE_SM100=1
export TORCH_CUDA_ARCH_LIST=9.0a

# Upstream FA3 is used for BF16 attention.
python -m pip install \
  https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.02.17-06dc5e7/flash_attn_3-3.0.0+20260216.cu128torch290cxx11abitrue.b62d93-cp39-abi3-linux_x86_64.whl

# FP8 KV attention uses vLLM's FA3 fork with two-level accumulation.
bash "${SCRIPT_DIR}/install_fp8_kv_backend.sh"

python -m pip install \
  flashinfer-python==0.6.12 \
  flashinfer-cubin==0.6.12
python -m pip install \
  flashinfer-jit-cache==0.6.12 \
  --index-url https://flashinfer.ai/whl/cu129

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

git clone --filter=blob:none https://github.com/deepseek-ai/FlashMLA.git \
  "${WORK_DIR}/FlashMLA"
git -C "${WORK_DIR}/FlashMLA" checkout 1408756
git -C "${WORK_DIR}/FlashMLA" submodule update --init --recursive
python -m pip install -v "${WORK_DIR}/FlashMLA" --no-build-isolation

export TORCH_CUDA_ARCH_LIST=9.0
export NVSHMEM_DIR="${NVSHMEM_DIR:-${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvshmem}"
if [[ -e "${NVSHMEM_DIR}/lib/libnvshmem_host.so.3" ]]; then
  ln -sfn "${NVSHMEM_DIR}/lib/libnvshmem_host.so.3" \
    "${NVSHMEM_DIR}/lib/libnvshmem_host.so"
fi

git clone --filter=blob:none https://github.com/deepseek-ai/DeepEP.git \
  "${WORK_DIR}/DeepEP"
git -C "${WORK_DIR}/DeepEP" checkout 567632d
(
  cd "${WORK_DIR}/DeepEP"
  python setup.py install
)

git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git \
  "${WORK_DIR}/DeepGEMM"
git -C "${WORK_DIR}/DeepGEMM" checkout \
  559d79fb6994a58b8a15b4b93bf13ccc16edf247
git -C "${WORK_DIR}/DeepGEMM" submodule update --init --recursive
(
  cd "${WORK_DIR}/DeepGEMM"
  ./install.sh
  ./develop.sh
)

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/pipeswift.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export FLASHINFER_CUDA_ARCH_LIST="9.0a"
export FLASH_ATTN_CUDA_ARCHS=90
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/pipeswift.sh" <<'EOF'
unset CUDA_HOME
unset CUDA_PATH
unset FLASHINFER_CUDA_ARCH_LIST
unset FLASH_ATTN_CUDA_ARCHS
unset CPATH
EOF

mkdir -p "${ROOT_DIR}/_deps"
if [[ ! -d "${ROOT_DIR}/_deps/cutlass/.git" ]]; then
  rm -rf "${ROOT_DIR}/_deps/cutlass"
  git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
    "${ROOT_DIR}/_deps/cutlass"
fi
git -C "${ROOT_DIR}/_deps/cutlass" fetch origin
git -C "${ROOT_DIR}/_deps/cutlass" checkout --detach f3fde583

export TORCH_CUDA_ARCH_LIST=9.0a
python -m pip install -e . --no-build-isolation -v

echo "PipeSwift build completed successfully."
