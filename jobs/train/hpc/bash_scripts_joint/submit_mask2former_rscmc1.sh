#!/usr/bin/env bash
set -euo pipefail

# Submit Mask2Former Swin-T and HRNet-W32 models on RSCMC1 with reference-style optimization.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_mask2former_rscmc1.sh [MODELS] [BATCH] [EPOCHS] [SEED] [DRY_RUN]
#
# MODELS: all (default), swin, hrnet, or a comma-separated combination.

MODELS="${1:-all}"
BATCH="${2:-4}"
EPOCHS="${3:-100}"
SEED="${4:-0}"
DRY_RUN="${5:-0}"

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
IMGSZ="${IMGSZ:-448}"
MULTISPECTRAL="${MULTISPECTRAL:-003}"
WORKERS="${WORKERS:-1}"
PROJECT="${PROJECT:-model_comparison_rscmc1_segment}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)-RSCMC1}"

OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.0001}"
LRF="${LRF:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
BACKBONE_LR_MULTIPLIER="${BACKBONE_LR_MULTIPLIER:-0.1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.01}"

JOBS=(
  "swin|m2f_swin|ultralytics/cfg/models/transformer/mask2former-swin-timm-seg.yaml"
  "hrnet|m2f_hr32|ultralytics/cfg/models/transformer/mask2former-hrnet-w32-timm-seg.yaml"
)

[[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -ge 1 ]] || { echo "BATCH must be an integer >= 1"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

selected() {
  local alias="$1"
  [[ "$MODELS" == "all" || ",${MODELS}," == *",${alias},"* ]]
}

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submitted=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r alias job_name config <<< "$entry"
  if ! selected "$alias"; then
    continue
  fi
  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  varlist="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_mask2former_${alias},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=0,SEED=${SEED},WANDB=true,PLOTS=false,PROJECT=${PROJECT},USE_SOFT_IGNORE=false,SDICE=1,OPTIMIZER=${OPTIMIZER},LR0=${LR0},LRF=${LRF},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_EPOCHS=${WARMUP_EPOCHS},BACKBONE_LR_MULTIPLIER=${BACKBONE_LR_MULTIPLIER},GRAD_CLIP_NORM=${GRAD_CLIP_NORM}${NO_AUG}"

  echo "Submitting Mask2Former ${alias}: imgsz=${IMGSZ}, batch=${BATCH}, lr=${LR0}, backbone_lr_multiplier=${BACKBONE_LR_MULTIPLIER}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
  submitted=$((submitted + 1))
done

[[ "$submitted" -gt 0 ]] || { echo "No models selected. Use all, swin, hrnet, or swin,hrnet."; exit 1; }
echo "Processed ${submitted} Mask2Former job(s)."
