#!/usr/bin/env bash
set -euo pipefail

# DINOv3 + YOLO11x dataset ablation at fixed image size 448.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_dataset_ablation.sh [TASK] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [DRY_RUN]
#
# TASK: all (default), obb, or segment

TASK_SET="${1:-all}"
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"
DRY_RUN="${6:-0}"

IMGSZ=448
PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
WORKERS="${WORKERS:-1}"

case "$TASK_SET" in all|obb|segment) ;; *) echo "TASK must be all, obb, or segment"; exit 1 ;; esac
[[ "$BATCH_OBB" =~ ^[0-9]+$ && "$BATCH_OBB" -ge 1 ]] || { echo "BATCH_OBB must be an integer >= 1"; exit 1; }
[[ "$BATCH_SEG" =~ ^[0-9]+$ && "$BATCH_SEG" -ge 1 ]] || { echo "BATCH_SEG must be an integer >= 1"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

# tag|multispectral_code|class_count
DATASETS=(
  "RSC|0|1"
  "RSCMC1|003|1"
  "RSCMC2|001|2"
  "PNSC|2|1"
  "PNSCMC1|21|1"
)

config_for() {
  local task="$1"
  local classes="$2"
  if [[ "$task" == "obb" ]]; then
    echo "ultralytics/cfg/models/timm/obb/final/augfpn/transformer/dinov3_7_12_17_22/${classes}cls/dinov3_7_12_17_22-augfpn_512c-obb.yaml"
  else
    echo "ultralytics/cfg/models/timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/${classes}cls/dinov3_7_12_17_22-yolo11x-segment.yaml"
  fi
}

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submit_job() {
  local task="$1"
  local tag="$2"
  local multispectral="$3"
  local classes="$4"
  local batch project prefix config task_args varlist job_name

  config="$(config_for "$task" "$classes")"
  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  if [[ "$task" == "obb" ]]; then
    batch="$BATCH_OBB"
    project="dino_dataset_ablation_obb"
    prefix="o"
    task_args=",ANGLE_MODE=le90"
  else
    batch="$BATCH_SEG"
    project="dino_dataset_ablation_segment"
    prefix="s"
    task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  fi

  job_name="ab_dino_${prefix}_${tag}"
  varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${job_name},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${multispectral},BATCH=${batch},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=1,SEED=${SEED},WANDB=true,PROJECT=${project}${task_args}${NO_AUG}"

  echo "Submitting ${job_name}: task=${task}, dataset=${tag}, config=${config}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

for dataset in "${DATASETS[@]}"; do
  IFS='|' read -r tag multispectral classes <<< "$dataset"
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "obb" ]] && submit_job obb "$tag" "$multispectral" "$classes"
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "segment" ]] && submit_job segment "$tag" "$multispectral" "$classes"
done
