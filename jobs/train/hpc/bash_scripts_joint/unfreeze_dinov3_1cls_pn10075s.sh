#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Submit joint freeze/unfreeze sweeps for the final 1-class DINOv3 models on PN10075S.
# This submits both:
#   - OBB: DINOv3 + AugFPN
#   - SEG: DINOv3 + YOLO11x neck
#
# Sweep settings:
#   - FREEZE=1
#   - LORA=false
#   - UNFREEZE in {4, 3, 2, 1}
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/unfreeze_dinov3_1cls_pn10075s.sh [IMGSZ] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]

IMGSZ="${1:-448}"
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MS_CODE="21"
MS_TAG="PN10075S"

OBB_CONFIG="$(timm_obb_final_augfpn_config dinov3_7_12_17_22 1cls)"
SEG_CONFIG="$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 1cls)"

OBB_PROJECT="unfreeze_obb_dinov3_1cls_pn10075s"
SEG_PROJECT="unfreeze_seg_dinov3_yolo11x_1cls_pn10075s"

FREEZE_VALUE=1
LORA_VALUE=false
UNFREEZE_VALUES=(4 3 2 1)

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
}

submit_job() {
  local task="$1"
  local unfreeze="$2"

  local config_yaml batch project label_prefix extra_params
  case "${task}" in
    obb)
      config_yaml="${OBB_CONFIG}"
      batch="${BATCH_OBB}"
      project="${OBB_PROJECT}"
      label_prefix="obb_dino"
      extra_params=",CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
      ;;
    segment)
      config_yaml="${SEG_CONFIG}"
      batch="${BATCH_SEG}"
      project="${SEG_PROJECT}"
      label_prefix="seg_dy11"
      extra_params=",CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25"
      ;;
    *)
      echo "[ERROR] Unsupported task: ${task}" >&2
      exit 1
      ;;
  esac

  local ts
  ts="$(date +%m%d-%H%M%S)"

  local label="${label_prefix}_f${FREEZE_VALUE}_u${unfreeze}_${MS_TAG}_c${IMGSZ}"
  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${FREEZE_VALUE},UNFREEZE=${unfreeze},LORA=${LORA_VALUE},EXPERIMENT_MODE=${ts}_${label},CONFIG_YAML=${config_yaml}${extra_params}"

  echo "→ ${label} | task=${task} freeze=${FREEZE_VALUE} unfreeze=${unfreeze} lora=${LORA_VALUE}"
  qsub -V -v "${varlist}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

require_file "${PBS_SCRIPT}"
require_file "${OBB_CONFIG}"
require_file "${SEG_CONFIG}"

TOTAL=$(( ${#UNFREEZE_VALUES[@]} * 2 ))
COUNT=0

echo "Submitting ${TOTAL} joint freeze/unfreeze jobs"
echo "IMGSZ=${IMGSZ} | MS_CODE=${MS_CODE} (${MS_TAG}) | BATCH_OBB=${BATCH_OBB} | BATCH_SEG=${BATCH_SEG} | EPOCHS=${EPOCHS} | SEED=${SEED}"
echo "Freeze=${FREEZE_VALUE} | LoRA=${LORA_VALUE} | Unfreeze values: ${UNFREEZE_VALUES[*]}"
echo "OBB config: ${OBB_CONFIG}"
echo "SEG config: ${SEG_CONFIG}"

for unfreeze in "${UNFREEZE_VALUES[@]}"; do
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "obb" "${unfreeze}"

  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "${unfreeze}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
