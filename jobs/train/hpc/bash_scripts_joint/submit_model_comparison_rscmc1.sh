#!/usr/bin/env bash
set -euo pipefail

# Model comparison on RSCMC1: OBB and segmentation at 448 by default.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_model_comparison_rscmc1.sh [MODELS] [DRY_RUN]
#
# MODELS: all (default), obb, segment, or comma-separated aliases from --list
#
# Examples:
#   bash jobs/train/hpc/bash_scripts_joint/submit_model_comparison_rscmc1.sh
#   bash jobs/train/hpc/bash_scripts_joint/submit_model_comparison_rscmc1.sh pointrend,mask2former,rhino
#   bash jobs/train/hpc/bash_scripts_joint/submit_model_comparison_rscmc1.sh --list

MODELS="${1:-all}"
DRY_RUN="${2:-0}"

export IMGSZ_OBB="${IMGSZ_OBB:-448}"
export IMGSZ_SEG="${IMGSZ_SEG:-448}"
export MULTISPECTRAL=003
export DATASET_NAME="RSCMC1"
export PROJECT_OBB="${PROJECT_OBB:-model_comparison_rscmc1_imgsz${IMGSZ_OBB}_obb}"
export PROJECT_SEG="${PROJECT_SEG:-model_comparison_rscmc1_imgsz${IMGSZ_SEG}_segment}"
export RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)-RSCMC1}"

BATCH_OBB="${BATCH_OBB:-8}"
BATCH_SEG="${BATCH_SEG:-8}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"

exec bash jobs/train/hpc/bash_scripts_joint/submit_benchmark_models.sh \
  "$MODELS" "$BATCH_OBB" "$BATCH_SEG" "$EPOCHS" "$SEED" "$DRY_RUN"
