#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../bash_scripts_joint/timm_paths.sh"

# Usage: bash backbone_sweep_loop.sh [IMGSZ] [BATCH]
IMGSZ="${1:-448}"
MULTISPECTRAL="${2:-0}"
BATCH="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"


# Basic validation
[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }
# [[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -ge 1  ]] || { echo "BATCH must be positive integer"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
CLS_TAG="1cls"
case "${MULTISPECTRAL}" in
  001) CLS_TAG="2cls" ;;
  002) CLS_TAG="4cls" ;;
esac

# Common env (task/config family stays segment_yolo_neck; tweak if you switch task)
COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25,WANDB=true,PROJECT=backbone_sweep_segment_final_dice1,SEED=${SEED}"

seg_yolo_cfg() {
  timm_segment_final_yolo_neck_config "$1" "$2" "${CLS_TAG}" flat
}

submit_job() {
  local label="$1" config_yaml="$2" freeze="$3"
  local ts; ts="$(date +%m%d-%H%M%S)"
  local VARLIST="${COMMON_PARAMS},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${label},CONFIG_YAML=${config_yaml}"
  echo "→ ${label}  [FREEZE=${freeze}]"
  qsub -V -v "${VARLIST}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

# Format: "label|model|head|freeze"
JOBS=(
  # # CNN BACKBONES — ResNet family

  # # CNN BACKBONES — EfficientNet family

  # # CNN BACKBONES — YOLO family
  "yolo12x_yolo12x|yolo12x|yolo12x|0"

  # # CNN BACKBONES — ConvNeXt family

  # # CNN BACKBONES — RegNet family

  # # CNN BACKBONES — MobileNet family

  # # CNN BACKBONES — DenseNet family

  # # CNN BACKBONES — HRNet family
  "hrnet_w32_rgb_yolo11_neck|hrnet_w32|yolo11x|0"
  "hrnet_w32_rgb_yolo12_neck|hrnet_w32|yolo12x_mlp2|0"

  # # VISION TRANSFORMERS — Swin family

  # # VISION TRANSFORMERS — PVT family

  # # VISION TRANSFORMERS — DeiT family

  # # VISION TRANSFORMERS — BEiT family

  # # VISION TRANSFORMERS — MaxViT family

  # # VISION TRANSFORMERS — EfficientViT family

  # # VISION TRANSFORMERS — MobileViT family

  # # HYBRID — CoAtNet family

  # # FOUNDATION — DINOv3 (frozen backbone)
  "dinov3_vitb14_rgb_yolo11_neck|dinov3_vitb14|yolo11x|1"
  "dinov3_vitb14_rgb_yolo12_neck|dinov3_vitb14|yolo12x|1"
)

# Summary
ACTIVE_COUNT=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  ACTIVE_COUNT=$((ACTIVE_COUNT+1))
done

echo "Submitting ${ACTIVE_COUNT} jobs  [IMGSZ=${IMGSZ}, BATCH=${BATCH}, MULTISPECTRAL=${MULTISPECTRAL}]"

i=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  IFS='|' read -r label model head freeze <<< "${job}"
  config_yaml="$(seg_yolo_cfg "${model}" "${head}")"
  timm_require_file "${config_yaml}"
  i=$((i+1))
  printf '[%02d/%02d] ' "${i}" "${ACTIVE_COUNT}"
  submit_job "${label}" "${config_yaml}" "${freeze}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
