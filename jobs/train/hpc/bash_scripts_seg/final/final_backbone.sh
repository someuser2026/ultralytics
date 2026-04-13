#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../bash_scripts_joint/timm_paths.sh"

# Usage: bash backbone_sweep_loop.sh [IMGSZ] [BATCH]
IMGSZ="${1:-448}"
BATCH="${2:-8}"
SEED="${3:-0}"
EPOCHS="${4:-100}"


# Basic validation
[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }
# [[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -ge 1  ]] || { echo "BATCH must be positive integer"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MULTISPECTRAL=0

# Common env (task/config family stays segment_no_p2; tweak if you switch task)
COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25,WANDB=true,PROJECT=backbone_sweep_segment_final_dice1,SEED=${SEED}"

seg_cfg() {
  timm_segment_final_panet_config "$1" no_p2 1cls
}

submit_job() {
  local label="$1" config_yaml="$2" freeze="$3"
  local ts; ts="$(date +%m%d-%H%M%S)"
  local VARLIST="${COMMON_PARAMS},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${label},CONFIG_YAML=${config_yaml}"
  echo "→ ${label}  [FREEZE=${freeze}]"
  qsub -V -v "${VARLIST}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

# Format: "label|model|freeze"
JOBS=(
  # CNN BACKBONES — ResNet family
  "resnet50_rgb|resnet50|0"
  "resnext50_32x4d_rgb|resnext50_32x4d|0"
  # "resnest50d_rgb|resnest50d|0"
  "seresnet50_rgb|seresnet50|0"
  # "skresnext50_32x4d_rgb|skresnext50_32x4d|0"

  # CNN BACKBONES — EfficientNet family
  "efficientnet_b1_rgb|efficientnet_b1|0"
  # "efficientnet_b3_rgb|efficientnet_b3|0"
  "efficientnetv2_s_rgb|efficientnetv2_s|0"

  # CNN BACKBONES — YOLO family
  "yolov11x|yolo11x|0"
  "yolo12x|yolo12x|0"

  # CNN BACKBONES — ConvNeXt family
  "convnext_small_rgb|convnext_small|0"
  "convnext_base_rgb|convnext_base|0"

  # CNN BACKBONES — RegNet family
  "regnety_040_rgb|regnety_040|0"

  # CNN BACKBONES — MobileNet family

  # CNN BACKBONES — DenseNet family
  "densenet121_rgb|densenet121|0"

  # CNN BACKBONES — HRNet family
  "hrnet_w32_rgb|hrnet_w32|0"

  # VISION TRANSFORMERS — Swin family
  "swin_tiny_rgb|swin_tiny_patch4_window7_224|0"

  # VISION TRANSFORMERS — PVT family
  "pvt_tiny_rgb|pvt_tiny|0"

  # VISION TRANSFORMERS — DeiT family
  "deit_small_rgb|deit_small_p16_224|0"

  # VISION TRANSFORMERS — BEiT family

  # VISION TRANSFORMERS — MaxViT family
  "maxvit_small_rgb|maxvit_small|0"

  # VISION TRANSFORMERS — EfficientViT family

  # VISION TRANSFORMERS — MobileViT family
  "mobilevit_s_rgb|mobilevit_s|0"

  # HYBRID — CoAtNet family
  "coatnet_0_rgb|coatnet_0|0"

  # FOUNDATION — DINOv3 (frozen backbone)
  "dinov3_vitb14_rgb|dinov3_vitb14|1"
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
  IFS='|' read -r label model freeze <<< "${job}"
  config_yaml="$(seg_cfg "${model}")"
  timm_require_file "${config_yaml}"
  i=$((i+1))
  printf '[%02d/%02d] ' "${i}" "${ACTIVE_COUNT}"
  submit_job "${label}" "${config_yaml}" "${freeze}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
