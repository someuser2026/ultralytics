#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# 4-run augmentation sweep:
#   - OBB DINOv3:   no-aug vs aug
#   - SEG DINO+Y11: no-aug vs aug
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/augmentation_on_off_sweep_dino.sh <IMGSZ> <MS_CODE> [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]

IMGSZ="${1:-448}"
MS_CODE="${2:-0}"
BATCH_OBB="${3:-8}"
BATCH_SEG="${4:-8}"
EPOCHS="${5:-100}"
SEED="${6:-0}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

resolve_ms() {
  local opt="$1"
  case "$opt" in
    0|RGB100|rgb100)          MS_CODE="0";   MS_TAG="RGB100";    CLS_TAG="1cls" ;;
    003|RGB10075S|rgb10075s)  MS_CODE="003"; MS_TAG="RGB10075S"; CLS_TAG="1cls" ;;
    001|RGB10075M|rgb10075m)  MS_CODE="001"; MS_TAG="RGB10075M"; CLS_TAG="2cls" ;;
    002|RGBALL|rgball)        MS_CODE="002"; MS_TAG="RGBALL";    CLS_TAG="4cls" ;;
    2|PN100|pn100)            MS_CODE="2";   MS_TAG="PN100";     CLS_TAG="1cls" ;;
    21|PN10075S|pn10075s)     MS_CODE="21";  MS_TAG="PN10075S";  CLS_TAG="1cls" ;;
    *)
      echo "[ERROR] Unsupported MS_CODE: $opt"
      echo "Allowed: 0, 003, 001, 002, 2, 21"
      exit 1 ;;
  esac
}

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
  qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

resolve_ms "${MS_CODE}"

OBB_CFG="$(timm_obb_final_augfpn_config dinov3_7_12_17_22 "${CLS_TAG}")"
SEG_CFG="$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x "${CLS_TAG}")"
require_file "${OBB_CFG}"
require_file "${SEG_CFG}"

PROJECT_OBB="aug_sweep_obb_dinov3_${MS_TAG}_c${IMGSZ}"
PROJECT_SEG="aug_sweep_seg_dinov3_y11_${MS_TAG}_c${IMGSZ}"

# No-aug profiles
OBB_NOAUG=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_NOAUG=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,SDICE=1,GAUSSIAN_BLUR_P=0.0"

# Aug profiles (kept from your previous settings)
OBB_AUG=",CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_AUG=",CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25"

echo "Submitting 4 jobs"
echo "Fixed setup: IMGSZ=${IMGSZ}, MS_CODE=${MS_CODE} (${MS_TAG}), CLS_TAG=${CLS_TAG}"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

submit_job "obb" "obb_dino_noaug_${MS_TAG}_c${IMGSZ}" "obb_dino_noaug_${MS_TAG}_c${IMGSZ}" "${OBB_CFG}" "1" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_NOAUG}"
submit_job "obb" "obb_dino_aug_${MS_TAG}_c${IMGSZ}" "obb_dino_aug_${MS_TAG}_c${IMGSZ}" "${OBB_CFG}" "1" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_AUG}"
submit_job "segment" "seg_dy11_noaug_${MS_TAG}_c${IMGSZ}" "seg_dy11_noaug_${MS_TAG}_c${IMGSZ}" "${SEG_CFG}" "1" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_NOAUG}"
submit_job "segment" "seg_dy11_aug_${MS_TAG}_c${IMGSZ}" "seg_dy11_aug_${MS_TAG}_c${IMGSZ}" "${SEG_CFG}" "1" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_AUG}"

echo "Done. Monitor with: qstat -u \"$USER\""
