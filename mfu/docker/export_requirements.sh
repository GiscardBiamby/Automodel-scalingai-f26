#!/usr/bin/env bash
# Regenerate mfu/docker/requirements-cu128.txt from uv.lock.
# Keeps every locked version for the extras the Nano-V3 recipe needs, and drops:
#   - torch/torchvision/triton and all CUDA-13 runtime wheels (nvidia-*, cuda-*): torch cu128 brings its own
#     nvidia-*-cu12 wheels; nvidia-cudnn-frontend (CUDA-agnostic) is re-added
#   - packages compiled from source in Dockerfile.cu128 (TE, mamba-ssm, causal-conv1d, DeepEP)
#   - CUDA-13-only packages not used by the recipe (tilelang, tile-kernels, apache-tvm-ffi, onnxruntime-gpu)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv export --frozen --no-hashes --no-emit-project --no-header --no-annotate --extra cuda --extra moe --extra tracking \
  | grep -vE '^nvidia-[a-z0-9-]+==' | cat - <(echo "nvidia-cudnn-frontend==1.29.0") \
  | grep -vE '^(torch|torchvision|triton|cuda-bindings|cuda-python|cuda-pathfinder|transformer-engine[a-z0-9-]*|mamba-ssm|causal-conv1d|deep-ep|tilelang|tile-kernels|apache-tvm-ffi|torch-c-dlpack-ext|onnxruntime-gpu)[ =@]' \
  | grep -vE "sys_platform == 'darwin'|sys_platform != 'linux'" \
  > mfu/docker/requirements-cu128.txt
wc -l mfu/docker/requirements-cu128.txt
