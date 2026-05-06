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
  BATCH       Optional YOLO inference batch size for directory predict mode
  JOB_BATCH_SIZE
              Optional image count per submitted PBS batch job; 0 or unset keeps single-job submission
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
JOB_BATCH_SIZE="${JOB_BATCH_SIZE:-0}"
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
[[ "${JOB_BATCH_SIZE}" =~ ^[0-9]+$ ]] || { echo "JOB_BATCH_SIZE must be a non-negative integer"; exit 1; }
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

resolve_image_root() {
  local scratch_root="$1"
  local site_name="$2"
  local img_dir="$3"
  local part
  if [[ "${img_dir}" = /* ]]; then
    echo "IMG_DIR must be relative under \$SCRATCH/data_processed/<site>/PSScene: ${img_dir}" >&2
    exit 1
  fi
  IFS='/' read -r -a img_parts <<< "${img_dir}"
  for part in "${img_parts[@]}"; do
    if [[ "${part}" == ".." ]]; then
      echo "IMG_DIR must not traverse outside PSScene: ${img_dir}" >&2
      exit 1
    fi
  done
  printf '%s/data_processed/%s/PSScene/%s\n' "${scratch_root}" "${site_name}" "${img_dir}"
}

list_pngs() {
  local image_root="$1"
  find "${image_root}" -maxdepth 1 -type f \( -iname '*.png' \) | sort
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
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}"; exit 1; }
[[ -f "${REPO_ROOT}/${PBS_SCRIPT}" ]] || { echo "PBS script not found: ${REPO_ROOT}/${PBS_SCRIPT}"; exit 1; }

RUN_NAME="$(infer_run_name "${CHECKPOINT}")"
JOB_LABEL="${JOB_LABEL:-$(sanitize_label "${RUN_NAME}_${SITE_NAME}")}"

echo "============================================================"
echo "Submitting site-level YOLO JSON prediction job"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Site: ${SITE_NAME}"
echo "  Image dir: ${IMG_DIR}"
echo "  Image size: ${IMGSZ}"
echo "  Confidence: ${CONF}"
echo "  IoU: ${IOU}"
echo "  Max det: ${MAX_DET:-auto}"
echo "  Inference batch: ${BATCH:-auto}"
echo "  Job batch size: ${JOB_BATCH_SIZE}"
echo "  Device: ${DEVICE}"
echo "  Predict mode: ${PREDICT_MODE}"
echo "  W&B upload: ${WANDB}"
echo "  Job label: ${JOB_LABEL}"
echo "============================================================"

BASE_VARS=(
  "CHECKPOINT=${CHECKPOINT}"
  "SITE_NAME=${SITE_NAME}"
  "IMG_DIR=${IMG_DIR}"
  "IMGSZ=${IMGSZ}"
  "CONF=${CONF}"
  "IOU=${IOU}"
  "DEVICE=${DEVICE}"
  "PREDICT_MODE=${PREDICT_MODE}"
)
if [[ -n "${MAX_DET}" ]]; then
  BASE_VARS+=("MAX_DET=${MAX_DET}")
fi
if [[ -n "${BATCH}" ]]; then
  BASE_VARS+=("BATCH=${BATCH}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  BASE_VARS+=("WANDB_RUN_ID=${WANDB_RUN_ID}")
fi

if [[ "${JOB_BATCH_SIZE}" == "0" ]]; then
  VARS=("${BASE_VARS[@]}" "WANDB=${WANDB}")
  submit_cmd "${JOB_LABEL}" "${VARS[@]}"
  exit 0
fi

[[ -n "${SCRATCH:-}" ]] || { echo "SCRATCH must be set when JOB_BATCH_SIZE > 0"; exit 1; }
[[ "${JOB_BATCH_SIZE}" -gt 0 ]] || { echo "JOB_BATCH_SIZE must be > 0 for batched submission"; exit 1; }

IMAGE_ROOT="$(resolve_image_root "${SCRATCH}" "${SITE_NAME}" "${IMG_DIR}")"
[[ -d "${IMAGE_ROOT}" ]] || { echo "Image directory not found: ${IMAGE_ROOT}"; exit 1; }
PNG_FILES=()
while IFS= read -r png_path; do
  PNG_FILES+=("${png_path}")
done < <(list_pngs "${IMAGE_ROOT}")
TOTAL_IMAGES="${#PNG_FILES[@]}"
[[ "${TOTAL_IMAGES}" -gt 0 ]] || { echo "No PNG images found in ${IMAGE_ROOT}"; exit 1; }

TOTAL_BATCHES=$(( (TOTAL_IMAGES + JOB_BATCH_SIZE - 1) / JOB_BATCH_SIZE ))
echo "Resolved image root: ${IMAGE_ROOT}"
echo "Found PNG images: ${TOTAL_IMAGES}"
echo "Submitting PBS batches: ${TOTAL_BATCHES}"

for ((batch_index=1; batch_index<=TOTAL_BATCHES; batch_index++)); do
  batch_start=$(( (batch_index - 1) * JOB_BATCH_SIZE ))
  batch_end=$(( batch_start + JOB_BATCH_SIZE ))
  if [[ "${batch_end}" -gt "${TOTAL_IMAGES}" ]]; then
    batch_end="${TOTAL_IMAGES}"
  fi

  batch_job_label="${JOB_LABEL}_b${batch_index}"
  echo "  Batch ${batch_index}: images [${batch_start}:${batch_end}) -> ${batch_job_label}"
  VARS=(
    "${BASE_VARS[@]}"
    "WANDB=false"
    "JOB_BATCH_INDEX=${batch_index}"
    "JOB_BATCH_START=${batch_start}"
    "JOB_BATCH_END=${batch_end}"
  )
  submit_cmd "${batch_job_label}" "${VARS[@]}"
done
