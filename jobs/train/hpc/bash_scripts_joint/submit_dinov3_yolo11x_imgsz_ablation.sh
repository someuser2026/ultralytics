#!/usr/bin/env bash
set -euo pipefail

# DINOv3 + YOLO11x image-size ablation.
# OBB uses PNSCMC1; segmentation uses RSCMC1.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_imgsz_ablation.sh [TASK] [IMGSZ]
#
# TASK: all (default), obb, or segment
# IMGSZ: all (default), 224, 448, or 896
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_imgsz_ablation.sh
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_imgsz_ablation.sh segment 896

TASK_SET="${1:-all}"
IMGSZ_SET="${2:-all}"
BATCH_OBB="${BATCH_OBB:-8}"
BATCH_SEG="${BATCH_SEG:-8}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
WORKERS="${WORKERS:-1}"
IMGSIZES=(224 448 896)

case "$TASK_SET" in all|obb|segment) ;; *) echo "TASK must be all, obb, or segment"; exit 1 ;; esac
case "$IMGSZ_SET" in all|224|448|896) ;; *) echo "IMGSZ must be all, 224, 448, or 896"; exit 1 ;; esac
[[ "$BATCH_OBB" =~ ^[0-9]+$ && "$BATCH_OBB" -ge 1 ]] || { echo "BATCH_OBB must be an integer >= 1"; exit 1; }
[[ "$BATCH_SEG" =~ ^[0-9]+$ && "$BATCH_SEG" -ge 1 ]] || { echo "BATCH_SEG must be an integer >= 1"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

OBB_CONFIG="ultralytics/cfg/models/timm/obb/final/augfpn/transformer/dinov3_7_12_17_22/1cls/dinov3_7_12_17_22-augfpn_512c-obb.yaml"
SEG_CONFIG="ultralytics/cfg/models/timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/dinov3_7_12_17_22-yolo11x-segment.yaml"

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submit_job() {
  local task="$1"
  local imgsz="$2"
  local batch project prefix config task_args multispectral dataset varlist job_name

  if [[ "$task" == "obb" ]]; then
    batch="$BATCH_OBB"
    project="dino_imgsz_ablation_obb"
    prefix="o"
    config="$OBB_CONFIG"
    multispectral="21"
    dataset="PNSCMC1"
    task_args=",ANGLE_MODE=le90"
  else
    batch="$BATCH_SEG"
    project="dino_imgsz_ablation_segment"
    prefix="s"
    config="$SEG_CONFIG"
    multispectral="003"
    dataset="RSCMC1"
    task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  fi

  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  job_name="ab_dino_${prefix}_${dataset}_c${imgsz}"
  varlist="TASK=${task},IMGSZ=${imgsz},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${job_name},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${multispectral},BATCH=${batch},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=1,SEED=${SEED},WANDB=true,PLOTS=false,PROJECT=${project}${task_args}${NO_AUG}"

  echo "Submitting ${job_name}: task=${task}, imgsz=${imgsz}, dataset=${dataset}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

for imgsz in "${IMGSIZES[@]}"; do
  [[ "$IMGSZ_SET" == "all" || "$IMGSZ_SET" == "$imgsz" ]] || continue
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "obb" ]] && submit_job obb "$imgsz"
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "segment" ]] && submit_job segment "$imgsz"
done
