#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Sweep models for each task with fixed dataset + tile size.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/model_sweep_fixed_data_tile.sh <IMGSZ> <MS_CODE> [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/model_sweep_fixed_data_tile.sh 448 0

IMGSZ="${1:-}"
MS_CODE="${2:-}"
BATCH_OBB="${3:-8}"
BATCH_SEG="${4:-8}"
EPOCHS="${5:-100}"
SEED="${6:-0}"

[[ -n "${IMGSZ}" ]] || { echo "[ERROR] IMGSZ is required"; exit 1; }
[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }
[[ -n "${MS_CODE}" ]] || { echo "[ERROR] MS_CODE is required"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

resolve_ms() {
  local opt="$1"
  case "$opt" in
    0|RGB100|rgb100)        MS_CODE="0";   MS_TAG="RGB100";    CLS_TAG="1cls" ;;
    003|RGB10075S|rgb10075s) MS_CODE="003"; MS_TAG="RGB10075S"; CLS_TAG="1cls" ;;
    001|RGB10075M|rgb10075m) MS_CODE="001"; MS_TAG="RGB10075M"; CLS_TAG="2cls" ;;
    002|RGBALL|rgball)      MS_CODE="002"; MS_TAG="RGBALL";    CLS_TAG="4cls" ;;
    2|PN100|pn100)          MS_CODE="2";   MS_TAG="PN100";     CLS_TAG="1cls" ;;
    21|PN10075S|pn10075s)   MS_CODE="21";  MS_TAG="PN10075S";  CLS_TAG="1cls" ;;
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

PROJECT_OBB="model_sweep_obb_${MS_TAG}_c${IMGSZ}"
PROJECT_SEG="model_sweep_seg_${MS_TAG}_c${IMGSZ}"

# Augmentations disabled for these runs.
OBB_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,SDICE=1,GAUSSIAN_BLUR_P=0.0"

# Format: label|config_yaml|freeze
OBB_MODELS=(
  "o_dino|$(timm_obb_final_augfpn_config dinov3_7_12_17_22 "${CLS_TAG}")|1"
  "o_hr32|$(timm_obb_final_augfpn_config hrnet_w32 "${CLS_TAG}")|0"
  "o_y11|$(timm_obb_final_augfpn_config yolo11x "${CLS_TAG}")|0"
  "o_y12|$(timm_obb_final_augfpn_config yolo12x "${CLS_TAG}")|0"
)

SEG_MODELS=(
  "s_dy11|$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x "${CLS_TAG}")|1"
  "s_hy11|$(timm_segment_final_yolo_neck_config hrnet_w32 yolo11x "${CLS_TAG}")|0"
  "s_y11|$(timm_segment_final_yolo_neck_config yolo11x yolo11x "${CLS_TAG}")|0"
  "s_y12|$(timm_segment_final_yolo_neck_config yolo12x yolo12x "${CLS_TAG}")|0"
)

for entry in "${OBB_MODELS[@]}"; do
  IFS='|' read -r _label cfg _freeze <<< "${entry}"
  require_file "${cfg}"
done
for entry in "${SEG_MODELS[@]}"; do
  IFS='|' read -r _label cfg _freeze <<< "${entry}"
  require_file "${cfg}"
done

TOTAL=$(( ${#OBB_MODELS[@]} + ${#SEG_MODELS[@]} ))
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "Fixed setup: IMGSZ=${IMGSZ}, MS_CODE=${MS_CODE} (${MS_TAG}), CLS_TAG=${CLS_TAG}"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

for entry in "${OBB_MODELS[@]}"; do
  IFS='|' read -r label cfg freeze <<< "${entry}"
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "obb" "${label}_${MS_TAG}_c${IMGSZ}" "${label}_${MS_TAG}_c${IMGSZ}" "${cfg}" "${freeze}" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_EXTRA}"
done

for entry in "${SEG_MODELS[@]}"; do
  IFS='|' read -r label cfg freeze <<< "${entry}"
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "${label}_${MS_TAG}_c${IMGSZ}" "${label}_${MS_TAG}_c${IMGSZ}" "${cfg}" "${freeze}" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_EXTRA}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
