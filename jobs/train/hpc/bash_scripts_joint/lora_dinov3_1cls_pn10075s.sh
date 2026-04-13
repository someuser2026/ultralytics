#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

# Submit joint LoRA sweeps for the final 1-class DINOv3 models on PN10075S.
# This submits both:
#   - OBB: DINOv3 + AugFPN
#   - SEG: DINOv3 + YOLO11x neck
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/lora_dinov3_1cls_pn10075s.sh [IMGSZ] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED]

IMGSZ="${1:-448}"
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "[ERROR] IMGSZ must be an integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MS_CODE="21"
MS_TAG="PN10075S"

OBB_CONFIG="$(timm_obb_final_augfpn_config dinov3_7_12_17_22 1cls)"
SEG_CONFIG="$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x 1cls)"

OBB_PROJECT="lora_obb_dinov3_1cls_pn10075s"
SEG_PROJECT="lora_seg_dinov3_yolo11x_1cls_pn10075s"

LORA_VALUES=(4 8 16)
LORA_DROPOUTS=(0.0 0.05)
LORA_LAYER_SPECS=(16 22)

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || { echo "[ERROR] Missing file: ${path}" >&2; exit 1; }
}

tagify_float() {
  printf '%s' "${1//./p}"
}

submit_job() {
  local task="$1"
  local rank="$2"
  local alpha="$3"
  local dropout="$4"
  local layers="$5"

  local config_yaml batch project label_prefix extra_params
  case "${task}" in
    obb)
      config_yaml="${OBB_CONFIG}"
      batch="${BATCH_OBB}"
      project="${OBB_PROJECT}"
      label_prefix="obb_dino"
      extra_params=",CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
      ;;
    segment)
      config_yaml="${SEG_CONFIG}"
      batch="${BATCH_SEG}"
      project="${SEG_PROJECT}"
      label_prefix="seg_dy11"
      extra_params=",CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25"
      ;;
    *)
      echo "[ERROR] Unsupported task: ${task}" >&2
      exit 1
      ;;
  esac

  local dropout_tag
  dropout_tag="$(tagify_float "${dropout}")"
  local ts
  ts="$(date +%m%d-%H%M%S)"

  local label="${label_prefix}_lora_r${rank}_a${alpha}_d${dropout_tag}_l${layers}_${MS_TAG}_c${IMGSZ}"
  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=1,EXPERIMENT_MODE=${ts}_${label},CONFIG_YAML=${config_yaml}${extra_params},LORA=true,LORA_RANK=${rank},LORA_ALPHA=${alpha},LORA_DROPOUT=${dropout},LORA_LAYERS=${layers}"

  echo "→ ${label} | task=${task} rank=${rank} alpha=${alpha} dropout=${dropout} layers=${layers}"
  qsub -V -v "${varlist}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

require_file "${PBS_SCRIPT}"
require_file "${OBB_CONFIG}"
require_file "${SEG_CONFIG}"

TOTAL=$(( ${#LORA_VALUES[@]} * ${#LORA_DROPOUTS[@]} * ${#LORA_LAYER_SPECS[@]} * 2 ))
COUNT=0

echo "Submitting ${TOTAL} joint LoRA jobs"
echo "IMGSZ=${IMGSZ} | MS_CODE=${MS_CODE} (${MS_TAG}) | BATCH_OBB=${BATCH_OBB} | BATCH_SEG=${BATCH_SEG} | EPOCHS=${EPOCHS} | SEED=${SEED}"
echo "LoRA values (rank=alpha): ${LORA_VALUES[*]} | Layers: ${LORA_LAYER_SPECS[*]}"
echo "OBB config: ${OBB_CONFIG}"
echo "SEG config: ${SEG_CONFIG}"

for value in "${LORA_VALUES[@]}"; do
  for dropout in "${LORA_DROPOUTS[@]}"; do
    for layers in "${LORA_LAYER_SPECS[@]}"; do
      COUNT=$((COUNT + 1))
      printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
      submit_job "obb" "${value}" "${value}" "${dropout}" "${layers}"

      COUNT=$((COUNT + 1))
      printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
      submit_job "segment" "${value}" "${value}" "${dropout}" "${layers}"
    done
  done
done

echo "Done. Monitor with: qstat -u \"$USER\""
