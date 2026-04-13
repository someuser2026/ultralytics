#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Cumulative augmentation ablation using DINOv3 for both tasks.
# Order: CLAHE -> +UNSHARP -> +GAUSSIAN_BLUR -> +MOTION_BLUR -> +MULTI_SPEC_NOISE
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/augmentation_ablation_cumulative_dino.sh <IMGSZ> <MS_CODE> [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [WALLTIME]
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/augmentation_ablation_cumulative_dino.sh 448 21 8 8 100 0 11:50:00

IMGSZ="${1:-448}"
MS_CODE="${2:-0}"
BATCH_OBB="${3:-8}"
BATCH_SEG="${4:-8}"
EPOCHS="${5:-100}"
SEED="${6:-0}"
WALLTIME="${7:-11:00:00}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

resolve_ms() {
  local opt="$1"
  case "$opt" in
    0|RGB100|rgb100)          MS_CODE="0";   MS_TAG="RGB100";    CLS_TAG="1cls" ;;
    01|RGBMA|rgbma)           MS_CODE="01";  MS_TAG="RGBMA";     CLS_TAG="1cls" ;;
    003|RGB10075S|rgb10075s)  MS_CODE="003"; MS_TAG="RGB10075S"; CLS_TAG="1cls" ;;
    001|RGB10075M|rgb10075m)  MS_CODE="001"; MS_TAG="RGB10075M"; CLS_TAG="2cls" ;;
    002|RGBALL|rgball)        MS_CODE="002"; MS_TAG="RGBALL";    CLS_TAG="4cls" ;;
    004|RGBALLSOFT|rgballsoft) MS_CODE="004"; MS_TAG="RGBALLSOFT"; CLS_TAG="1cls" ;;
    1|NIR|nir)                MS_CODE="1";   MS_TAG="NIR";       CLS_TAG="1cls" ;;
    11|NIRMA|nirma)           MS_CODE="11";  MS_TAG="NIRMA";     CLS_TAG="1cls" ;;
    2|PN100|pn100)            MS_CODE="2";   MS_TAG="PN100";     CLS_TAG="1cls" ;;
    21|PN10075S|pn10075s)     MS_CODE="21";  MS_TAG="PN10075S";  CLS_TAG="1cls" ;;
    3|NM100|nm100)            MS_CODE="3";   MS_TAG="NM100";     CLS_TAG="1cls" ;;
    *)
      echo "[ERROR] Unsupported MS_CODE: $opt"
      echo "Allowed: 0, 01, 001, 002, 003, 004, 1, 11, 2, 21, 3"
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
  qsub -V -l walltime="${WALLTIME}" -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

resolve_ms "${MS_CODE}"

OBB_CFG="$(timm_obb_final_augfpn_config dinov3_7_12_17_22 "${CLS_TAG}")"
SEG_CFG="$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x "${CLS_TAG}")"
require_file "${OBB_CFG}"
require_file "${SEG_CFG}"

PROJECT_OBB="aug_ablation_obb_dinov3_${MS_TAG}_c${IMGSZ}"
PROJECT_SEG="aug_ablation_seg_dinov3_y11_${MS_TAG}_c${IMGSZ}"

STAGE_NAMES=(
  "clahe"
  "clahe_unsharp"
  "clahe_unsharp_gblur"
  "clahe_unsharp_gblur_mblur"
  "clahe_unsharp_gblur_mblur_msnoise"
)

STAGE_CLAHE=(0.25 0.25 0.25 0.25 0.25)
STAGE_UNSHARP=(0.0 0.50 0.50 0.50 0.50)
STAGE_GBLUR=(0.0 0.0 0.25 0.25 0.25)
STAGE_MBLUR=(0.0 0.0 0.0 0.50 0.50)
STAGE_MSNOISE=(0.0 0.0 0.0 0.0 0.10)

TOTAL=$(( ${#STAGE_NAMES[@]} * 2 ))
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "Fixed setup: IMGSZ=${IMGSZ}, MS_CODE=${MS_CODE} (${MS_TAG}), CLS_TAG=${CLS_TAG}, EPOCHS=${EPOCHS}, SEED=${SEED}, WALLTIME=${WALLTIME}"
echo "Ablation order: CLAHE -> +UNSHARP -> +GAUSSIAN_BLUR -> +MOTION_BLUR -> +MULTI_SPEC_NOISE"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

for idx in "${!STAGE_NAMES[@]}"; do
  stage_num=$((idx + 1))
  stage_name="${STAGE_NAMES[$idx]}"

  clahe="${STAGE_CLAHE[$idx]}"
  unsharp="${STAGE_UNSHARP[$idx]}"
  gblur="${STAGE_GBLUR[$idx]}"
  mblur="${STAGE_MBLUR[$idx]}"
  msnoise="${STAGE_MSNOISE[$idx]}"

  aug_common=",CLAHE_P=${clahe},UNSHARP_P=${unsharp},GAUSSIAN_BLUR_P=${gblur},MOTION_BLUR_P=${mblur},MULTI_SPEC_NOISE_P=${msnoise}"

  obb_extra="${aug_common}"
  seg_extra="${aug_common},SDICE=1"

  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "obb" "obb_dino_augab${stage_num}_${MS_TAG}_c${IMGSZ}" "obb_dino_${stage_name}_${MS_TAG}_c${IMGSZ}" "${OBB_CFG}" "1" "${BATCH_OBB}" "${PROJECT_OBB}" "${obb_extra}"

  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "seg_dy11_augab${stage_num}_${MS_TAG}_c${IMGSZ}" "seg_dy11_${stage_name}_${MS_TAG}_c${IMGSZ}" "${SEG_CFG}" "1" "${BATCH_SEG}" "${PROJECT_SEG}" "${seg_extra}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
