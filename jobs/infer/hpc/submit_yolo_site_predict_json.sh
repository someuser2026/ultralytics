#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PBS_SCRIPT="jobs/infer/hpc/yolo_site_predict_json.pbs"

usage() {
  cat <<'EOF'
Usage:
  bash jobs/infer/hpc/submit_yolo_site_predict_json.sh CHECKPOINT SITE_NAME IMG_DIR [IMGSZ] [CONF] [DEVICE] [DRY_RUN]

Example:
  bash jobs/infer/hpc/submit_yolo_site_predict_json.sh \
    /srv/scratch/.../best.pt \
    Treachery \
    visual/pngs/images_c448_ov35_kf20 \
    640 0.25 0 1

Positional arguments:
  CHECKPOINT  Path to checkpoint weights (.pt)
  SITE_NAME   Site name under $SCRATCH/data_processed/<site_name>
  IMG_DIR     Relative path under $SCRATCH/data_processed/<site_name>/PSScene
  IMGSZ       Optional inference image size (default: 448)
  CONF        Optional confidence threshold (default: 0.01)
  DEVICE      Optional device (default: 0)
  DRY_RUN     Optional 0/1 flag (default: 0)

Optional environment overrides:
  IOU         IoU threshold for NMS (default: 0.45)
  MAX_DET     Maximum detections per image (default in Python script: 100 for segment, 300 otherwise)
  BATCH       Optional batch size for directory predict mode
  PREDICT_MODE Either `per-image` or `directory` (default: per-image)
  WANDB       Whether to upload the prediction directory to W&B (default: true)
  WANDB_RUN_ID Optional W&B run ID to resume instead of creating a sibling inference run
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CHECKPOINT_INPUT="${1:-}"
SITE_NAME="${2:-}"
IMG_DIR="${3:-}"
IMGSZ="${4:-448}"
CONF="${5:-0.01}"
DEVICE="${6:-0}"
DRY_RUN="${7:-${DRY_RUN:-0}}"
IOU="${IOU:-0.45}"
MAX_DET="${MAX_DET:-}"
BATCH="${BATCH:-}"
PREDICT_MODE="${PREDICT_MODE:-per-image}"
WANDB="${WANDB:-true}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

[[ -n "${CHECKPOINT_INPUT}" ]] || { echo "CHECKPOINT is required"; usage; exit 1; }
[[ -n "${SITE_NAME}" ]] || { echo "SITE_NAME is required"; usage; exit 1; }
[[ -n "${IMG_DIR}" ]] || { echo "IMG_DIR is required"; usage; exit 1; }
[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "IMGSZ must be an integer >= 32"; exit 1; }
[[ "${CONF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "CONF must be numeric"; exit 1; }
[[ "${IOU}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "IOU must be numeric"; exit 1; }
[[ -n "${DEVICE}" ]] || { echo "DEVICE must not be empty"; exit 1; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -z "${MAX_DET}" || "${MAX_DET}" =~ ^[0-9]+$ ]] || { echo "MAX_DET must be empty or a non-negative integer"; exit 1; }
[[ -z "${BATCH}" || "${BATCH}" =~ ^[0-9]+$ ]] || { echo "BATCH must be empty or a non-negative integer"; exit 1; }
[[ "${PREDICT_MODE}" == "per-image" || "${PREDICT_MODE}" == "directory" ]] || { echo "PREDICT_MODE must be 'per-image' or 'directory'"; exit 1; }
[[ "${WANDB}" == "true" || "${WANDB}" == "false" ]] || { echo "WANDB must be 'true' or 'false'"; exit 1; }

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
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}"; exit 1; }
[[ -f "${REPO_ROOT}/${PBS_SCRIPT}" ]] || { echo "PBS script not found: ${REPO_ROOT}/${PBS_SCRIPT}"; exit 1; }

RUN_NAME="$(infer_run_name "${CHECKPOINT}")"
JOB_LABEL="${JOB_LABEL:-$(sanitize_label "${RUN_NAME}_${SITE_NAME}")}"

VARS=(
  "CHECKPOINT=${CHECKPOINT}"
  "SITE_NAME=${SITE_NAME}"
  "IMG_DIR=${IMG_DIR}"
  "IMGSZ=${IMGSZ}"
  "CONF=${CONF}"
  "IOU=${IOU}"
  "DEVICE=${DEVICE}"
  "PREDICT_MODE=${PREDICT_MODE}"
  "WANDB=${WANDB}"
)
if [[ -n "${MAX_DET}" ]]; then
  VARS+=("MAX_DET=${MAX_DET}")
fi
if [[ -n "${BATCH}" ]]; then
  VARS+=("BATCH=${BATCH}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  VARS+=("WANDB_RUN_ID=${WANDB_RUN_ID}")
fi
VARLIST="$(IFS=,; echo "${VARS[*]}")"
CMD=(qsub -V -v "${VARLIST}" -N "${JOB_LABEL}" "${PBS_SCRIPT}")

echo "============================================================"
echo "Submitting site-level YOLO JSON prediction job"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Site: ${SITE_NAME}"
echo "  Image dir: ${IMG_DIR}"
echo "  Image size: ${IMGSZ}"
echo "  Confidence: ${CONF}"
echo "  IoU: ${IOU}"
echo "  Max det: ${MAX_DET:-auto}"
echo "  Batch: ${BATCH:-auto}"
echo "  Device: ${DEVICE}"
echo "  Predict mode: ${PREDICT_MODE}"
echo "  W&B upload: ${WANDB}"
echo "  Job label: ${JOB_LABEL}"
echo "============================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
