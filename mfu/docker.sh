#!/usr/bin/env bash
# Run a command inside the Automodel container (default: automodel-mfu:cu128, see mfu/docker/Dockerfile.cu128) with this checkout mounted over /opt/Automodel.
#
# Usage:
#   mfu/docker.sh [--name CONTAINER_NAME] [-- CMD ...]     (no CMD -> interactive bash)
#
# - Runs as the host user so files written to mfu/runs/ are not root-owned.
# - Mounts the host HF cache (~/.cache/huggingface) and a per-user cache dir for triton/torch/wandb.
# - Forwards WANDB_API_KEY / HF_TOKEN from the host environment when set (never written to disk here).
# - MFU_GPUS=2,3,4,5,6,7 exposes only those host GPUs (default: all).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AUTOMODEL_IMAGE:-automodel-mfu:cu128}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
USER_CACHE="${MFU_CACHE:-$HOME/.cache/automodel-mfu}"
NAME="automodel-mfu-${USER}-$(date +%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
mkdir -p "$HF_CACHE" "$USER_CACHE/home"
GPU_SPEC="all"
[[ -n "${MFU_GPUS:-}" ]] && GPU_SPEC="\"device=${MFU_GPUS}\""

TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)
[[ $# -eq 0 ]] && set -- bash

exec docker run --rm "${TTY_FLAGS[@]}" \
  --name "$NAME" \
  --gpus "$GPU_SPEC" \
  --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache/home \
  -e USER="$USER" -e LOGNAME="$USER" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e WANDB_DIR=/cache/wandb \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF \
  -e PYTHONFAULTHANDLER \
  -e NCCL_DEBUG \
  -e CUDA_DEVICE_MAX_CONNECTIONS \
  -v "$REPO_DIR":/opt/Automodel \
  -v "$HF_CACHE":/hf \
  -v "$USER_CACHE":/cache \
  -w /opt/Automodel \
  "$IMAGE" "$@"
