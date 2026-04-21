#!/usr/bin/env bash
set -euo pipefail

# Joint Mamba model launcher for the fixed single-class Planet+Nearmap dataset.
# Fixed setup:
#   - IMGSZ=448
#   - MULTISPECTRAL=21 (PN10075S)
#   - OBB: Mamba-YOLO-L and HRNet-Mamba
#   - SEG: YOLO-Mamba and HRNet-Mamba
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/mamba_models_448_pn10075s.sh [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [DRY_RUN]
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/mamba_models_448_pn10075s.sh 8 8 100 0 0

IMGSZ=448
MS_CODE="21"
MS_TAG="PN10075S"
MS_DESC="Planet+Nearmap (100+75 collapsed, single class)"

BATCH_OBB="${1:-8}"
BATCH_SEG="${2:-8}"
EPOCHS="${3:-80}"
SEED="${4:-0}"
DRY_RUN="${5:-0}"

PBS_SCRIPT="jobs/train/hpc/planet_full_3hr_walltime.pbs"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[ERROR] Missing config: $path" >&2; exit 1; }
}

submit_job() {
  local task="$1"
  local job_name="$2"
  local exp_suffix="$3"
  local config_yaml="$4"
  local freeze="$5"
  local batch="$6"
  local project="$7"
  local extra_params="$8"

  local ts
  ts="$(date +%m%d-%H%M%S)"

  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${exp_suffix},CONFIG_YAML=${config_yaml}${extra_params}"

  echo "→ ${job_name} | task=${task} | cfg=${config_yaml}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN qsub -V -v \"${varlist}\" -N \"${job_name}\" \"${PBS_SCRIPT}\""
  else
    qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
    sleep 1
  fi
}

PROJECT_OBB="mamba_joint_obb_${MS_TAG}_c${IMGSZ}"
PROJECT_SEG="mamba_joint_seg_${MS_TAG}_c${IMGSZ}"

OBB_EXTRA=",ANGLE_MODE=le90,CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
SEG_EXTRA=",CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25"

OBB_MODELS=(
  "o_mamba_l|ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml|0"
  "o_mamba_hr|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb.yaml|0"
  "o_mamba_l_edgevss|ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo-edgevss.yaml|0"
  "o_mamba_hr_edgevss|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb-edgevss.yaml|0"
)

SEG_MODELS=(
  "s_mamba|ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml|0"
  "s_mamba_hr|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|0"
  "s_mamba_edgevss_bb|ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg-edgevss-backbone.yaml|0"
  "s_mamba_edgevss_all|ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg-edgevss-all.yaml|0"
  "s_mamba_hr_edgevss|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg-edgevss.yaml|0"
)

for entry in "${OBB_MODELS[@]}"; do
  IFS='|' read -r _label cfg _freeze <<< "${entry}"
  require_file "${cfg}"
done
for entry in "${SEG_MODELS[@]}"; do
  IFS='|' read -r _label cfg _freeze <<< "${entry}"
  require_file "${cfg}"
done

TOTAL=$(( ${#OBB_MODELS[@]} + ${#SEG_MODELS[@]} ))
COUNT=0

echo "Submitting ${TOTAL} jobs"
echo "Fixed setup: IMGSZ=${IMGSZ}, MULTISPECTRAL=${MS_CODE} (${MS_TAG})"
echo "Dataset: ${MS_DESC}"
echo "EPOCHS=${EPOCHS}, SEED=${SEED}, BATCH_OBB=${BATCH_OBB}, BATCH_SEG=${BATCH_SEG}, DRY_RUN=${DRY_RUN}"
echo "WandB projects: ${PROJECT_OBB}, ${PROJECT_SEG}"

for entry in "${OBB_MODELS[@]}"; do
  IFS='|' read -r label cfg freeze <<< "${entry}"
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "obb" "${label}_${MS_TAG}_c${IMGSZ}" "${label}_${MS_TAG}_c${IMGSZ}" "${cfg}" "${freeze}" "${BATCH_OBB}" "${PROJECT_OBB}" "${OBB_EXTRA}"
done

for entry in "${SEG_MODELS[@]}"; do
  IFS='|' read -r label cfg freeze <<< "${entry}"
  COUNT=$((COUNT + 1))
  printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
  submit_job "segment" "${label}_${MS_TAG}_c${IMGSZ}" "${label}_${MS_TAG}_c${IMGSZ}" "${cfg}" "${freeze}" "${BATCH_SEG}" "${PROJECT_SEG}" "${SEG_EXTRA}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
