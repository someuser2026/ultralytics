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
  CONF        Optional confidence threshold (default: 0.001)
  DEVICE      Optional device (default: 0)
  DRY_RUN     Optional 0/1 flag (default: 0)
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
CONF="${5:-0.001}"
DEVICE="${6:-0}"
DRY_RUN="${7:-${DRY_RUN:-0}}"

[[ -n "${CHECKPOINT_INPUT}" ]] || { echo "CHECKPOINT is required"; usage; exit 1; }
[[ -n "${SITE_NAME}" ]] || { echo "SITE_NAME is required"; usage; exit 1; }
[[ -n "${IMG_DIR}" ]] || { echo "IMG_DIR is required"; usage; exit 1; }
[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "IMGSZ must be an integer >= 32"; exit 1; }
[[ "${CONF}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "CONF must be numeric"; exit 1; }
[[ -n "${DEVICE}" ]] || { echo "DEVICE must not be empty"; exit 1; }
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
  "DEVICE=${DEVICE}"
)
VARLIST="$(IFS=,; echo "${VARS[*]}")"
CMD=(qsub -V -v "${VARLIST}" -N "${JOB_LABEL}" "${PBS_SCRIPT}")

echo "============================================================"
echo "Submitting site-level YOLO JSON prediction job"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Site: ${SITE_NAME}"
echo "  Image dir: ${IMG_DIR}"
echo "  Image size: ${IMGSZ}"
echo "  Confidence: ${CONF}"
echo "  Device: ${DEVICE}"
echo "  Job label: ${JOB_LABEL}"
echo "============================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
