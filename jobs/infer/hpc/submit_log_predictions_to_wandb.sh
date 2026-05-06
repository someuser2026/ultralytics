#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PBS_SCRIPT="jobs/infer/hpc/log_predictions_to_wandb.pbs"

usage() {
  cat <<'EOF'
Usage:
  bash jobs/infer/hpc/submit_log_predictions_to_wandb.sh CHECKPOINT DATA [DEVICE] [BATCH] [IMGSZ] [CONF] [DRY_RUN]

Example:
  bash jobs/infer/hpc/submit_log_predictions_to_wandb.sh \
    /srv/scratch/.../best.pt \
    /srv/scratch/.../data.yaml \
    0 2 448 0.01 1

Positional arguments:
  CHECKPOINT   Path to checkpoint weights (.pt)
  DATA         Path to dataset YAML containing val/test splits
  DEVICE       Optional device (default: 0)
  BATCH        Optional batch size (default: use script/default behavior)
  IMGSZ        Optional inference image size (default: 448)
  CONF         Optional confidence threshold (default: 0.01)
  DRY_RUN      Optional 0/1 flag (default: 0)

Optional environment overrides:
  WANDB_RUN_ID Optional W&B run ID to resume instead of creating a sibling inference run
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CHECKPOINT_INPUT="${1:-}"
DATA_INPUT="${2:-}"
DEVICE="${3:-0}"
BATCH="${4:-}"
IMGSZ="${5:-448}"
CONF="${6:-0.01}"
DRY_RUN="${7:-${DRY_RUN:-0}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

[[ -n "${CHECKPOINT_INPUT}" ]] || { echo "CHECKPOINT is required"; usage; exit 1; }
[[ -n "${DATA_INPUT}" ]] || { echo "DATA is required"; usage; exit 1; }
[[ -n "${DEVICE}" ]] || { echo "DEVICE must not be empty"; exit 1; }
[[ -z "${BATCH}" || "${BATCH}" =~ ^[0-9]+$ ]] || { echo "BATCH must be empty or a non-negative integer"; exit 1; }
[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "IMGSZ must be an integer >= 32"; exit 1; }
[[ "${CONF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "CONF must be numeric"; exit 1; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    local dir base
    dir="$(dirname "${path}")"
    base="$(basename "${path}")"
    printf '%s/%s\n' "$(cd "${dir}" && pwd)" "${base}"
  fi
}

sanitize_label() {
  printf '%s' "$1" | tr '/[:space:]' '_' | tr -cd '[:alnum:]_.-'
}

join_by_comma() {
  local first="${1:-}"
  shift || true
  printf '%s' "${first}"
  local item
  for item in "$@"; do
    printf ',%s' "${item}"
  done
}

submit_cmd() {
  local job_name="$1"
  shift
  local varlist
  varlist="$(join_by_comma "$@")"
  local cmd=(qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY RUN] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

infer_run_name() {
  local ckpt="$1"
  local ckpt_name parent_name
  ckpt_name="$(basename "${ckpt}")"
  parent_name="$(basename "$(dirname "${ckpt}")")"
  if [[ "${ckpt_name}" == "best.pt" && "${parent_name}" == "weights" ]]; then
    basename "$(dirname "$(dirname "${ckpt}")")"
  else
    basename "${ckpt%.*}"
  fi
}

CHECKPOINT="$(resolve_path "${CHECKPOINT_INPUT}")"
DATA="$(resolve_path "${DATA_INPUT}")"
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}"; exit 1; }
[[ -f "${DATA}" ]] || { echo "Data YAML not found: ${DATA}"; exit 1; }
[[ -f "${REPO_ROOT}/${PBS_SCRIPT}" ]] || { echo "PBS script not found: ${REPO_ROOT}/${PBS_SCRIPT}"; exit 1; }

RUN_NAME="$(infer_run_name "${CHECKPOINT}")"
DATA_NAME="$(basename "$(dirname "${DATA}")")"
JOB_LABEL="${JOB_LABEL:-$(sanitize_label "${RUN_NAME}_${DATA_NAME}_valtest")}"

echo "============================================================"
echo "Submitting val/test W&B prediction export job"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Data YAML: ${DATA}"
echo "  Device: ${DEVICE}"
echo "  Batch: ${BATCH:-auto}"
echo "  Image size: ${IMGSZ}"
echo "  Confidence: ${CONF}"
echo "  W&B run id: ${WANDB_RUN_ID:-auto}"
echo "  Job label: ${JOB_LABEL}"
echo "============================================================"

VARS=(
  "CHECKPOINT=${CHECKPOINT}"
  "DATA=${DATA}"
  "DEVICE=${DEVICE}"
  "IMGSZ=${IMGSZ}"
  "CONF=${CONF}"
)
if [[ -n "${BATCH}" ]]; then
  VARS+=("BATCH=${BATCH}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  VARS+=("WANDB_RUN_ID=${WANDB_RUN_ID}")
fi

submit_cmd "${JOB_LABEL}" "${VARS[@]}"
