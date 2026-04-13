#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Sweep dataset variants at configurable tile size.
# Fixed models:
#   - OBB: DINOv3
#   - SEG: DINOv3 + YOLO11 neck
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/dataset_type_sweep_dino.sh [IMGSZ] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]

IMGSZ="${1:-448}"
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
PROJECT_OBB="dataset_type_sweep_obb_dinov3_c${IMGSZ}"
PROJECT_SEG="dataset_type_sweep_seg_dinov3_yolo11_c${IMGSZ}"

# Requested dataset sweep set (all dataset variants for this sweep).
DATASETS=(
  "RGB100|0|1cls"
  "RGB10075S|003|1cls"
  "RGB10075M|001|2cls"
  "PN100|2|1cls"
  "PN10075S|21|1cls"
)

obb_cfg_for_cls() {
  case "$1" in
    1cls) echo "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 1cls)" ;;
    2cls) echo "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 2cls)" ;;
    *) echo "[ERROR] Unsupported OBB cls tag: $1" >&2; exit 1 ;;
  esac
}

seg_cfg_for_cls() {
  case "$1" in
    1cls) echo "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 1cls)" ;;
    2cls) echo "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 2cls)" ;;
    *) echo "[ERROR] Unsupported SEG cls tag: $1" >&2; exit 1 ;;
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
  local multispectral="$6"
  local batch="$7"
  local project="$8"
  local extra_params="$9"

  local ts
  ts="$(date +%m%d-%H%M%S)"

  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${multispectral},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${exp_suffix},CONFIG_YAML=${config_yaml}${extra_params}"

  echo "→ ${job_name} | task=${task} | MS=${multispectral} | cfg=${config_yaml}"
  qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

# Augmentations disabled for these runs.
OBB_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,SDICE=1,GAUSSIAN_BLUR_P=0.0"

# Segment runs only for this retry.
TOTAL=${#DATASETS[@]}
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "IMGSZ=${IMGSZ} | EPOCHS=${EPOCHS} | SEED=${SEED}"
echo "WandB project: ${PROJECT_SEG}"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r dataset_tag ms_code cls_tag <<< "${entry}"

  obb_cfg="$(obb_cfg_for_cls "${cls_tag}")"
  seg_cfg="$(seg_cfg_for_cls "${cls_tag}")"
  require_file "${obb_cfg}"
  require_file "${seg_cfg}"

  # OBB retries disabled.
  # COUNT=$((COUNT + 1))
  # printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  # submit_job "obb" "obb_dino_${dataset_tag}_c${IMGSZ}" "obb_dino_${dataset_tag}_c${IMGSZ}" "${obb_cfg}" "1" "${ms_code}" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_EXTRA}"

  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "seg_dy11_${dataset_tag}_c${IMGSZ}" "seg_dy11_${dataset_tag}_c${IMGSZ}" "${seg_cfg}" "1" "${ms_code}" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_EXTRA}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
