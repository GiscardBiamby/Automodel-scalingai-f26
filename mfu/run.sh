#!/usr/bin/env bash
# Launch one Nemotron-Nano-V3 MFU-study run inside the NGC Automodel container.
#
# Usage:
#   mfu/run.sh [options] [-- extra automodel overrides, e.g. --step_scheduler.local_batch_size=4]
#
# Options:
#   -c, --config PATH     recipe YAML (default: mfu/configs/nemotron_nano_v3_squad_baseline.yaml)
#   -n, --name NAME       run name, used for the run dir and W&B (default: config basename)
#   -s, --steps N         override step_scheduler.max_steps
#   -p, --profile MODE    none (default) | nsys
#       --nsys-steps A:B  optimizer-step capture window [A, B) for nsys (default 10:13)
#       --gpus N          processes per node (default 8, or the number of --devices)
#       --devices LIST    host GPU ids to use, e.g. 2,3,4,5,6,7 (default: all); other GPUs are not touched
#       --no-wandb        disable W&B logging
#       --force           skip the "GPUs are busy" guard (>4 GiB used or >10% util; shared machine!)
#
# Every run writes mfu/runs/<timestamp>_<name>/ containing: command.txt, git.txt (commit + diff),
# config.yaml (as launched), train.log, training.jsonl (per-step metrics) and, for nsys, profile.nsys-rep.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${AUTOMODEL_IMAGE:-automodel-mfu:cu128}"
CONFIG="mfu/configs/nemotron_nano_v3_squad_baseline.yaml"
NAME=""
STEPS=""
PROFILE="none"
NSYS_STEPS="10:13"
NPROC=""
DEVICES=""
WANDB=1
FORCE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG="$2"; shift 2 ;;
    -n|--name) NAME="$2"; shift 2 ;;
    -s|--steps) STEPS="$2"; shift 2 ;;
    -p|--profile) PROFILE="$2"; shift 2 ;;
    --nsys-steps) NSYS_STEPS="$2"; shift 2 ;;
    --gpus) NPROC="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --no-wandb) WANDB=0; shift ;;
    --force) FORCE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (automodel overrides go after --)" >&2; exit 2 ;;
  esac
done

cd "$REPO_DIR"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
[[ "$PROFILE" == "none" || "$PROFILE" == "nsys" ]] || { echo "--profile must be none|nsys" >&2; exit 2; }
NAME="${NAME:-$(basename "$CONFIG" .yaml)}"
if [[ -n "$DEVICES" ]]; then
  NPROC="${NPROC:-$(echo "$DEVICES" | tr ',' '\n' | wc -l)}"
  export MFU_GPUS="$DEVICES"
fi
NPROC="${NPROC:-8}"

# --- Shared-machine guard: refuse to start if anyone is using the GPUs ---------------------------
if [[ "$FORCE" -eq 0 ]]; then
  # Busy = >4 GiB used or >10% utilization in any of 3 samples (idle notebooks holding a little memory are OK).
  busy=$(for _ in 1 2 3; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 1; done \
    | awk -F', ' -v devs=",${DEVICES}," '(devs == ",," || index(devs, "," $1 ",")) && ($2 > 4096 || $3 > 10) {print $1}' | sort -u)
  if [[ -n "$busy" ]]; then
    echo "GPUs busy (>4 GiB used or >10% util): $(echo $busy | tr '\n' ' ')" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    docker ps --format '{{.Names}}\t{{.Status}}' | grep -i automodel >&2 || true
    echo "Refusing to start; wait for them to finish or pass --force." >&2
    exit 3
  fi
fi

# --- Run directory + provenance ------------------------------------------------------------------
RUN_ID="$(date +%Y%m%d-%H%M%S)_${NAME}"
RUN_DIR="mfu/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
{
  echo "commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "image:  $IMAGE"
  echo "gpus:   ${DEVICES:-all} (nproc $NPROC)"
  echo "host:   $(hostname)"
  echo; git status --short; echo; git diff HEAD
} > "$RUN_DIR/git.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"

OVERRIDES=(
  "--checkpoint.checkpoint_dir=/opt/Automodel/${RUN_DIR}"
  "--wandb.name=${RUN_ID}"
)
[[ -n "$STEPS" ]] && OVERRIDES+=("--step_scheduler.max_steps=${STEPS}")
[[ "$WANDB" -eq 0 ]] && OVERRIDES+=("--wandb.enable=false")

LAUNCH=(torchrun --standalone --nproc-per-node "$NPROC" -m nemo_automodel.cli.app "/opt/Automodel/${RUN_DIR}/config.yaml")
if [[ "$PROFILE" == "nsys" ]]; then
  NSYS_START="${NSYS_STEPS%%:*}"
  NSYS_END="${NSYS_STEPS##*:}"
  OVERRIDES+=("--nvtx=true" "--profiling.nsys_start_step=${NSYS_START}" "--profiling.nsys_end_step=${NSYS_END}")
  LAUNCH=(nsys profile
    --output "/opt/Automodel/${RUN_DIR}/profile"
    --trace cuda,nvtx
    --capture-range cudaProfilerApi --capture-range-end stop
    --trace-fork-before-exec true
    --cuda-memory-usage false
    --sample none --cpuctxsw none
    --force-overwrite true
    "${LAUNCH[@]}")
fi
CMD=("${LAUNCH[@]}" "${OVERRIDES[@]}" "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"; echo >> "$RUN_DIR/command.txt"

# --- Launch --------------------------------------------------------------------------------------
if [[ "$WANDB" -eq 1 && -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc 2>/dev/null || true)"
  export WANDB_API_KEY
fi
echo "Run dir: $RUN_DIR"
echo "Command: $(cat "$RUN_DIR/command.txt")"
set +e
"$REPO_DIR/mfu/docker.sh" --name "automodel-mfu-${USER}-${RUN_ID//[^a-zA-Z0-9_.-]/-}" -- "${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
set -e
echo "exit status: $status" | tee -a "$RUN_DIR/train.log"
if [[ -f "$RUN_DIR/training.jsonl" ]]; then
  python3 "$REPO_DIR/mfu/summarize.py" "$RUN_DIR" | tee "$RUN_DIR/summary.md"
fi
exit "$status"
