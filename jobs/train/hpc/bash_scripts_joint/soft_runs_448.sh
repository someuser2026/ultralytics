#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Joint soft-dataset submission script (fixed setup):
#   - IMGSZ=448 only
#   - MULTISPECTRAL=004 only (planet_full ... all_soft ...)
#   - One mode only:
#       * OBB: DINOv3
#       * SEG: DINOv3 + YOLO11 neck
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/soft_runs_448.sh [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [DRY_RUN]
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/soft_runs_448.sh 8 8 100 0 0

IMGSZ=448
MS_CODE="004"
MS_TAG="RGBALLSOFT"
CLS_TAG="1cls"
MS_DESC="Planet RGB all_soft dataset (single class + per-annotation probability)"

BATCH_OBB="${1:-8}"
BATCH_SEG="${2:-8}"
EPOCHS="${3:-100}"
SEED="${4:-0}"
DRY_RUN="${5:-0}"

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[ERROR] Missing config: $path" >&2; exit 1; }
}

submit_job() {
  local task="$1"
  local job_name="$2"
  local exp_suffix="$3"
  local config_yaml="$4"
  local freeze="$5"
  local batch="$6"
  local project="$7"
  local extra_params="$8"

  local ts
  ts="$(date +%m%d-%H%M%S)"

  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${exp_suffix},CONFIG_YAML=${config_yaml}${extra_params}"

  echo "→ ${job_name} | task=${task} | cfg=${config_yaml}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN qsub -V -v \"${varlist}\" -N \"${job_name}\" \"${PBS_SCRIPT}\""
  else
    qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
    sleep 1
  fi
}

PROJECT_OBB="soft_runs_448_obb_dino_${MS_TAG}"
PROJECT_SEG="soft_runs_448_seg_dino_y11_${MS_TAG}"

# Augmentations disabled for controlled soft-data runs.
OBB_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,SDICE=1,GAUSSIAN_BLUR_P=0.0"

OBB_LABEL="o_dino"
OBB_CFG="$(timm_obb_final_augfpn_config dinov3_7_12_17_22 "${CLS_TAG}")"
OBB_FREEZE="1"

SEG_LABEL="s_dy11"
SEG_CFG="$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x "${CLS_TAG}")"
SEG_FREEZE="1"

require_file "${OBB_CFG}"
require_file "${SEG_CFG}"

TOTAL=2
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "Fixed setup: IMGSZ=${IMGSZ}, MULTISPECTRAL=${MS_CODE} (${MS_TAG}), CLS_TAG=${CLS_TAG}"
echo "Dataset: ${MS_DESC}"
echo "EPOCHS=${EPOCHS}, SEED=${SEED}, BATCH_OBB=${BATCH_OBB}, BATCH_SEG=${BATCH_SEG}, DRY_RUN=${DRY_RUN}"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

COUNT=$((COUNT + 1))
printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
submit_job "obb" "${OBB_LABEL}_${MS_TAG}_c${IMGSZ}" "${OBB_LABEL}_${MS_TAG}_c${IMGSZ}" "${OBB_CFG}" "${OBB_FREEZE}" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_EXTRA}"

COUNT=$((COUNT + 1))
printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
submit_job "segment" "${SEG_LABEL}_${MS_TAG}_c${IMGSZ}" "${SEG_LABEL}_${MS_TAG}_c${IMGSZ}" "${SEG_CFG}" "${SEG_FREEZE}" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_EXTRA}"

echo "Done. Monitor with: qstat -u \"$USER\""
