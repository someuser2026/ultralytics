#!/usr/bin/env bash
set -euo pipefail

# DINOv3 + YOLO11x image-size ablation across every benchmark dataset.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_all_dataset_imgsz_ablation.sh [TASK] [DATASET] [IMGSZ]
#
# TASK: all (default), obb, or segment
# DATASET: all (default), RSC, RSCMC1, RSCMC2, PNSC, or PNSCMC1
# IMGSZ: all (default), 224, 448, or 896
#
# Examples:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_all_dataset_imgsz_ablation.sh
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_all_dataset_imgsz_ablation.sh segment RSCMC2 896

TASK_SET="${1:-all}"
DATASET_SET="${2:-all}"
IMGSZ_SET="${3:-all}"

BATCH_OBB="${BATCH_OBB:-8}"
BATCH_SEG="${BATCH_SEG:-8}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
WORKERS="${WORKERS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
IMGSIZES=(224 448 896)

# tag|multispectral_code|class_count
DATASETS=(
  "RSC|0|1"
  "RSCMC1|003|1"
  "RSCMC2|001|2"
  "PNSC|2|1"
  "PNSCMC1|21|1"
)

case "$TASK_SET" in all|obb|segment) ;; *) echo "TASK must be all, obb, or segment"; exit 1 ;; esac
case "$DATASET_SET" in all|RSC|RSCMC1|RSCMC2|PNSC|PNSCMC1) ;; *) echo "DATASET must be all, RSC, RSCMC1, RSCMC2, PNSC, or PNSCMC1"; exit 1 ;; esac
case "$IMGSZ_SET" in all|224|448|896) ;; *) echo "IMGSZ must be all, 224, 448, or 896"; exit 1 ;; esac
[[ "$BATCH_OBB" =~ ^[0-9]+$ && "$BATCH_OBB" -ge 1 ]] || { echo "BATCH_OBB must be an integer >= 1"; exit 1; }
[[ "$BATCH_SEG" =~ ^[0-9]+$ && "$BATCH_SEG" -ge 1 ]] || { echo "BATCH_SEG must be an integer >= 1"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

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
  local imgsz="$5"
  local batch project prefix config task_args varlist job_name dataset_slug

  config="$(config_for "$task" "$classes")"
  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }
  dataset_slug="$(printf '%s' "$tag" | tr '[:upper:]' '[:lower:]')"

  if [[ "$task" == "obb" ]]; then
    batch="$BATCH_OBB"
    project="dino_dataset_imgsz_ablation_${dataset_slug}_imgsz${imgsz}_obb"
    prefix="o"
    task_args=",ANGLE_MODE=le90"
  else
    batch="$BATCH_SEG"
    project="dino_dataset_imgsz_ablation_${dataset_slug}_imgsz${imgsz}_segment"
    prefix="s"
    task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  fi

  job_name="ab_dino_${prefix}_${tag}_c${imgsz}"
  varlist="TASK=${task},IMGSZ=${imgsz},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${job_name},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${multispectral},BATCH=${batch},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=1,SEED=${SEED},WANDB=true,PLOTS=false,PROJECT=${project}${task_args}${NO_AUG}"

  echo "Submitting ${job_name}: task=${task}, dataset=${tag}, imgsz=${imgsz}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

for dataset in "${DATASETS[@]}"; do
  IFS='|' read -r tag multispectral classes <<< "$dataset"
  [[ "$DATASET_SET" == "all" || "$DATASET_SET" == "$tag" ]] || continue

  for imgsz in "${IMGSIZES[@]}"; do
    [[ "$IMGSZ_SET" == "all" || "$IMGSZ_SET" == "$imgsz" ]] || continue
    [[ "$TASK_SET" == "all" || "$TASK_SET" == "obb" ]] && submit_job obb "$tag" "$multispectral" "$classes" "$imgsz"
    [[ "$TASK_SET" == "all" || "$TASK_SET" == "segment" ]] && submit_job segment "$tag" "$multispectral" "$classes" "$imgsz"
  done
done
