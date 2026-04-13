#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Sweep tile sizes for fixed models:
#   - OBB: DINOv3
#   - SEG: DINOv3 + YOLO11 neck
#
# Dataset selection comes from script 1; keep it blank unless you provide arg1.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/tile_size_sweep_dino.sh <MS_CODE> [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]
#
# MS_CODE expected from dataset sweep:
#   0 (RGB100), 003 (RGB10075S), 001 (RGB10075M), 002 (RGBALL), 2 (PN100), 21 (PN10075S)

MS_CODE="${1:-}"   # Intentionally blank by default; set from dataset selection.
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"

TILE_SIZES=(224 448 896)

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
    "")
      echo "[ERROR] MS_CODE is blank. Pick one dataset code from script 1 and pass it as arg1."
      exit 1 ;;
    *)
      echo "[ERROR] Unsupported MS_CODE: $opt"
      echo "Allowed: 0, 003, 001, 002, 2, 21"
      exit 1 ;;
  esac
}

obb_cfg_for_cls() {
  case "$1" in
    1cls) echo "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 1cls)" ;;
    2cls) echo "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 2cls)" ;;
    4cls) echo "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 4cls)" ;;
    *) echo "[ERROR] Unsupported OBB cls tag: $1" >&2; exit 1 ;;
  esac
}

seg_cfg_for_cls() {
  case "$1" in
    1cls) echo "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 1cls)" ;;
    2cls) echo "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 2cls)" ;;
    4cls) echo "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 4cls)" ;;
    *) echo "[ERROR] Unsupported SEG cls tag: $1" >&2; exit 1 ;;
  esac
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[ERROR] Missing config: $path" >&2; exit 1; }
}

submit_job() {
  local task="$1"
  local imgsz="$2"
  local job_name="$3"
  local exp_suffix="$4"
  local config_yaml="$5"
  local freeze="$6"
  local batch="$7"
  local project="$8"
  local extra_params="$9"

  local ts
  ts="$(date +%m%d-%H%M%S)"

  local varlist="TASK=${task},IMGSZ=${imgsz},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${exp_suffix},CONFIG_YAML=${config_yaml}${extra_params}"

  echo "→ ${job_name} | task=${task} | IMGSZ=${imgsz} | MS=${MS_TAG}"
  qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

resolve_ms "${MS_CODE}"

OBB_CFG="$(obb_cfg_for_cls "${CLS_TAG}")"
SEG_CFG="$(seg_cfg_for_cls "${CLS_TAG}")"
require_file "${OBB_CFG}"
require_file "${SEG_CFG}"

PROJECT_OBB="tile_size_sweep_obb_dinov3_${MS_TAG}"
PROJECT_SEG="tile_size_sweep_seg_dinov3_yolo11_${MS_TAG}"

# Augmentations disabled for these runs.
OBB_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.0,UNSHARP_P=0.0,SDICE=1,GAUSSIAN_BLUR_P=0.0"

TOTAL=$(( ${#TILE_SIZES[@]} * 2 ))
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "MS_CODE=${MS_CODE} (${MS_TAG}) | CLS_TAG=${CLS_TAG}"
echo "Tile sizes: ${TILE_SIZES[*]}"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

for imgsz in "${TILE_SIZES[@]}"; do
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "obb" "${imgsz}" "obb_dino_${MS_TAG}_c${imgsz}" "obb_dino_${MS_TAG}_c${imgsz}" "${OBB_CFG}" "1" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_EXTRA}"

  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "${imgsz}" "seg_dy11_${MS_TAG}_c${imgsz}" "seg_dy11_${MS_TAG}_c${imgsz}" "${SEG_CFG}" "1" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_EXTRA}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
