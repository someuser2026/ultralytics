#!/usr/bin/env bash
set -euo pipefail

# Resume incomplete RSCMC1 model-comparison runs.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/resume_model_comparison_rscmc1.sh [MODELS] [DRY_RUN]
#
# MODELS: all (default), obb, segment, or comma-separated aliases from --list
# Optional environment overrides: SCRATCH, RUNS_ROOT, PROFILE, TARGET_EPOCHS, SEED, WANDB,
# IMGSZ_OBB, IMGSZ_SEG, PROJECT_OBB, PROJECT_SEG

MODELS="${1:-all}"
DRY_RUN="${2:-${DRY_RUN:-0}}"
IMGSZ_OBB="${IMGSZ_OBB:-448}"
IMGSZ_SEG="${IMGSZ_SEG:-448}"
PROJECT_OBB="${PROJECT_OBB:-model_comparison_rscmc1_imgsz${IMGSZ_OBB}_obb}"
PROJECT_SEG="${PROJECT_SEG:-model_comparison_rscmc1_imgsz${IMGSZ_SEG}_segment}"
SEED="${SEED:-0}"

source jobs/train/hpc/bash_scripts_joint/resume_model_comparison_lib.sh
resume_model_comparison
