#!/usr/bin/env bash
set -euo pipefail

# Submit segmentation models being evaluated for architecture improvements.
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_arch_improvement_models.sh [MODELS] [BATCH] [EPOCHS] [SEED] [DRY_RUN]
#
# MODELS may be:
#   all                         submit all models (default)
#   segment                     submit all segmentation models
#   model1,model2               submit selected model aliases from --list
#
# Examples:
#   bash jobs/train/hpc/bash_scripts_joint/submit_arch_improvement_models.sh all 4 100 0 1
#   bash jobs/train/hpc/bash_scripts_joint/submit_arch_improvement_models.sh mamba_hr_pointrend 4 100 0 0
#   bash jobs/train/hpc/bash_scripts_joint/submit_arch_improvement_models.sh --list
#
# Optional environment overrides: IMGSZ, MULTISPECTRAL, PROJECT, WORKERS, WANDB, RUN_TAG

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MODELS="${1:-all}"
BATCH="${2:-4}"
EPOCHS="${3:-100}"
SEED="${4:-0}"
DRY_RUN="${5:-0}"

IMGSZ="${IMGSZ:-448}"
MULTISPECTRAL="${MULTISPECTRAL:-21}"
PROJECT="${PROJECT:-architecture_improvement_segment}"
WORKERS="${WORKERS:-1}"
WANDB="${WANDB:-true}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"

# alias|task|job_name|config|freeze|pointrend_mode
JOBS=(
  "mamba_hr_pointrend|segment|a_mhr_pr|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg-pointrend.yaml|0|joint"
  "mamba_hrv|segment|a_mhrv|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg-dvss.yaml|0|"
)

if [[ "$MODELS" == "--list" ]]; then
  echo "SEG: mamba_hr_pointrend mamba_hrv"
  exit 0
fi

[[ "$BATCH" =~ ^[0-9]+$ && "$BATCH" -ge 1 ]] || { echo "BATCH must be an integer >= 1"; exit 1; }
[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be an integer >= 32"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$WORKERS" =~ ^[0-9]+$ ]] || { echo "WORKERS must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

selected() {
  local alias="$1"
  local task="$2"
  [[ "$MODELS" == "all" || "$MODELS" == "$task" || ",${MODELS}," == *",${alias},"* ]]
}

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submit_job() {
  local alias="$1"
  local task="$2"
  local job_name="$3"
  local config="$4"
  local freeze="$5"
  local pointrend_mode="$6"
  local task_args varlist

  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  if [[ -n "$pointrend_mode" ]]; then
    task_args+=",POINTREND_MODE=${pointrend_mode}"
  fi

  varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${alias},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=${freeze},SEED=${SEED},WANDB=${WANDB},PLOTS=false,PROJECT=${PROJECT}${task_args}${NO_AUG}"

  echo "Submitting ${alias}: task=${task}, imgsz=${IMGSZ}, batch=${BATCH}, config=${config}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

submitted=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r alias task job_name config freeze pointrend_mode <<< "$entry"
  if selected "$alias" "$task"; then
    submit_job "$alias" "$task" "$job_name" "$config" "$freeze" "$pointrend_mode"
    submitted=$((submitted + 1))
  fi
done

[[ "$submitted" -gt 0 ]] || { echo "No runnable models selected. Use --list for valid aliases."; exit 1; }
echo "Processed ${submitted} architecture-improvement job(s)."
