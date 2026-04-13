#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../bash_scripts_joint/timm_paths.sh"

# Usage: bash backbone_sweep_seeded.sh [IMGSZ] [BATCH] [MULTISPECTRAL] [DRY_RUN]
IMGSZ="${1:-224}"
BATCH="${2:-16}"
MULTISPECTRAL="${3:-0}"
DRY_RUN="${4:-0}"

# Basic validation
[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
CLS_TAG="1cls"
case "${MULTISPECTRAL}" in
  001) CLS_TAG="2cls" ;;
  002) CLS_TAG="4cls" ;;
esac

# Seeds to run
SEEDS=(0 1 2 3 4)

# Common env (task/config family stays OBB; tweak if you switch task)
BASE_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0,WANDB=true,PROJECT=obb_seed_runs"

submit_job() {
  local label="$1" config_yaml="$2" freeze="$3" seed="$4"
  local ts; ts="$(date +%m%d-%H%M%S)"
  local job_name="${label}_seed${seed}"
  local VARLIST="${BASE_PARAMS},FREEZE=${freeze},SEED=${seed},EXPERIMENT_MODE=${ts}_${label}_seed${seed},CONFIG_YAML=${config_yaml}"
  echo "→ ${job_name}  [FREEZE=${freeze}, SEED=${seed}]"
  
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY RUN] qsub -V -v \"${VARLIST}\" -N \"${job_name}\" \"${PBS_SCRIPT}\""
  else
    qsub -V -v "${VARLIST}" -N "${job_name}" "${PBS_SCRIPT}"
    sleep 1
  fi
}

# Format: "label|model|freeze"
JOBS=(
  #   # CNN BACKBONES — ResNet family
  "resnet50_rgb|resnet50|0"
  "resnext50_32x4d_rgb|resnext50_32x4d|0"
  "seresnet50_rgb|seresnet50|0"

#   # CNN BACKBONES — EfficientNet family
  "efficientnetv2_s_rgb|efficientnetv2_s|0"

#   # CNN BACKBONES — YOLO family
  "yolov11x|yolo11x|0"
  "yolo12x|yolo12x|0"

#   # CNN BACKBONES — ConvNeXt family
  "convnext_base_rgb|convnext_base|0"

#   # CNN BACKBONES — RegNet family
  "regnety_040_rgb|regnety_040|0"

#   # CNN BACKBONES — MobileNet family

#   # CNN BACKBONES — DenseNet family
  "densenet121_rgb|densenet121|0"

#   # CNN BACKBONES — HRNet family
  "hrnet_w32_rgb|hrnet_w32|0"

#   # VISION TRANSFORMERS — Swin family
  "swin_tiny_rgb|swin_tiny_patch4_window7_224|0"

#   # VISION TRANSFORMERS — PVT family
  "pvt_tiny_rgb|pvt_tiny|0"

#   # VISION TRANSFORMERS — DeiT family
  "deit_small_rgb|deit_small_p16_224|0"

#   # VISION TRANSFORMERS — BEiT family

#   # VISION TRANSFORMERS — MaxViT family
  "maxvit_tiny_rgb|maxvit_small|0"

#   # VISION TRANSFORMERS — EfficientViT family

#   # VISION TRANSFORMERS — MobileViT family
  "mobilevit_s_rgb|mobilevit_s|0"

#   # HYBRID — CoAtNet family
  "coatnet_0_rgb|coatnet_0|0"

#   # FOUNDATION — DINOv3 (frozen backbone)
  "dinov3_vitb14_rgb|dinov3_vitb14|1"
)

# Count total jobs (models × seeds)
ACTIVE_MODELS=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  ACTIVE_MODELS=$((ACTIVE_MODELS+1))
done

TOTAL_JOBS=$((ACTIVE_MODELS * ${#SEEDS[@]}))

echo "========================================="
if [ "$DRY_RUN" = "1" ]; then
  echo "🔍 DRY RUN MODE - No jobs will be submitted"
  echo "========================================="
fi
echo "Submitting ${TOTAL_JOBS} jobs (${ACTIVE_MODELS} models × ${#SEEDS[@]} seeds)"
echo "IMGSZ=${IMGSZ}, BATCH=${BATCH}, MULTISPECTRAL=${MULTISPECTRAL}"
echo "SEEDS: ${SEEDS[*]}"
echo "========================================="

job_counter=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  IFS='|' read -r label model freeze <<< "${job}"
  config_yaml="$(timm_obb_final_augfpn_config "${model}" "${CLS_TAG}")"
  timm_require_file "${config_yaml}"
  
  # Submit job for each seed
  for seed in "${SEEDS[@]}"; do
    job_counter=$((job_counter+1))
    printf '[%03d/%03d] ' "${job_counter}" "${TOTAL_JOBS}"
    submit_job "${label}" "${config_yaml}" "${freeze}" "${seed}"
  done
done

echo "========================================="
if [ "$DRY_RUN" = "1" ]; then
  echo "🔍 DRY RUN completed. No jobs were submitted."
else
  echo "Done. Monitor with: qstat -u \"$USER\""
fi
echo "========================================="
