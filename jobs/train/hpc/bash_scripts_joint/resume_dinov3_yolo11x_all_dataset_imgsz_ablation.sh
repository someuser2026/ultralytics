#!/usr/bin/env bash
set -euo pipefail

# Resume incomplete DINOv3 + YOLO11x dataset/image-size ablations.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/resume_dinov3_yolo11x_all_dataset_imgsz_ablation.sh [TASK] [DATASET] [IMGSZ] [DRY_RUN]
#
# TASK: all (default), obb, or segment
# DATASET: all (default), RSC, RSCMC1, RSCMC2, PNSC, or PNSCMC1
# IMGSZ: all (default), 224, 448, or 896
# DRY_RUN: 0 (default) or 1
#
# Optional environment overrides: SCRATCH, RUNS_ROOT, PROFILE, TARGET_EPOCHS, SEED, WANDB,
# PROJECT_OBB, PROJECT_SEG

TASK_SET="${1:-all}"
DATASET_SET="${2:-all}"
IMGSZ_SET="${3:-all}"
DRY_RUN="${4:-${DRY_RUN:-0}}"
SEED="${SEED:-0}"
PROJECT_OBB="${PROJECT_OBB:-dino_dataset_imgsz_ablation_obb}"
PROJECT_SEG="${PROJECT_SEG:-dino_dataset_imgsz_ablation_segment}"
IMGSIZES=(224 448 896)
DATASETS=(RSC RSCMC1 RSCMC2 PNSC PNSCMC1)

case "$TASK_SET" in all|obb|segment) ;; *) echo "TASK must be all, obb, or segment"; exit 1 ;; esac
case "$DATASET_SET" in all|RSC|RSCMC1|RSCMC2|PNSC|PNSCMC1) ;; *) echo "DATASET must be all, RSC, RSCMC1, RSCMC2, PNSC, or PNSCMC1"; exit 1 ;; esac
case "$IMGSZ_SET" in all|224|448|896) ;; *) echo "IMGSZ must be all, 224, 448, or 896"; exit 1 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }

source jobs/train/hpc/bash_scripts_joint/resume_incomplete_runs_lib.sh
resume_init

for dataset in "${DATASETS[@]}"; do
  [[ "$DATASET_SET" == "all" || "$DATASET_SET" == "$dataset" ]] || continue
  for imgsz in "${IMGSIZES[@]}"; do
    [[ "$IMGSZ_SET" == "all" || "$IMGSZ_SET" == "$imgsz" ]] || continue
    if [[ "$TASK_SET" == "all" || "$TASK_SET" == "obb" ]]; then
      resume_consider_run obb "$imgsz" "$PROJECT_OBB" "ab_dino_o_${dataset}_c${imgsz}_seed${SEED}"
    fi
    if [[ "$TASK_SET" == "all" || "$TASK_SET" == "segment" ]]; then
      resume_consider_run segment "$imgsz" "$PROJECT_SEG" "ab_dino_s_${dataset}_c${imgsz}_seed${SEED}"
    fi
  done
done

resume_submit_selected
