#!/usr/bin/env bash
# Install the FP8 KV cache attention backend: vLLM's FlashAttention fork.
#
# --enable_fp8_kv needs vLLM's FA3 fork rather than the upstream flash_attn_3
# wheel installed alongside it, which lacks the SM90 two-level FP8 accumulation
# fix (vllm-project/flash-attention#104): on an H100 the attention output drifts
# -1% at 1k context, -25% at 16k and -67% at 128k, measured against an exact
# reference. check_fp8_kv_backend.py is that measurement as a pass/fail gate.
#
# vLLM ships the fork inside its own wheel, and in the releases we target that
# subpackage imports nothing from vLLM -- only torch. So we lift it out and
# re-emit it as a standalone wheel: no CUDA compilation, and no vLLM runtime
# dependency pulled into the environment. The one hard constraint is the torch
# C++ ABI, so the vLLM release must be built against this environment's torch.
#
# Usage: bash scripts/install_fp8_kv_backend.sh [vllm_version]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python -c "import torch" 2>/dev/null; then
  echo "error: cannot import torch with '$(command -v python)'." >&2
  echo "       Activate PipeSwift's environment first, e.g.:" >&2
  echo "         conda activate pipeswift" >&2
  exit 1
fi

# vLLM release -> the torch it is built against. All of these postdate
# flash-attention#104 (merged 2025-11-05), so all carry the fix; verified for
# 0.16.0, which pins flash-attention 188be165 (2025-12-11). check_backend.py
# re-checks empirically rather than trusting this table.
declare -A VLLM_FOR_TORCH=(
  ["2.9"]="0.16.0"
  ["2.10"]="0.19.1"
  ["2.11"]="0.26.0"
  ["2.13"]="0.31.0"
)

TORCH_MM="$(python -c 'import torch; print(".".join(torch.__version__.split(".")[:2]))')"
TORCH_FULL="$(python -c 'import torch; print(torch.__version__)')"

if [[ $# -ge 1 ]]; then
  VLLM_VERSION="$1"
else
  VLLM_VERSION="${VLLM_FOR_TORCH[$TORCH_MM]:-}"
  if [[ -z "$VLLM_VERSION" ]]; then
    echo "error: no known vLLM release pinned to torch $TORCH_MM.x" >&2
    echo "       find one whose requires_dist has torch==$TORCH_MM.* and pass it:" >&2
    echo "       bash $0 <vllm_version>" >&2
    exit 1
  fi
fi

echo "[fp8-kv] torch $TORCH_FULL -> lifting vllm_flash_attn out of vllm==$VLLM_VERSION"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pip download "vllm==${VLLM_VERSION}" --no-deps --only-binary :all: -d "$WORK/dl"
WHEEL="$(ls "$WORK"/dl/vllm-*.whl | head -1)"

python "$SCRIPT_DIR/repack_vllm_flash_attn.py" "$WHEEL" -o "$WORK/out"
pip install --force-reinstall --no-deps "$WORK"/out/vllm_flash_attn-*.whl

echo "[fp8-kv] verifying the installed kernel actually has the SM90 FP8 fix"
python "$SCRIPT_DIR/check_fp8_kv_backend.py"

echo "[fp8-kv] done -- PipeSwift can now be launched with --enable_fp8_kv"
