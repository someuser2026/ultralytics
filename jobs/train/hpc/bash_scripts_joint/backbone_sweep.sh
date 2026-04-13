#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  echo "Usage: bash jobs/train/hpc/bash_scripts_joint/backbone_sweep.sh <profile> [IMGSZ] [BATCH] [MULTISPECTRAL]"
  echo "Profiles: segment_no_p2, segment_no_p2_stoc_drop, obb_base, obb_stoc_drop"
  exit 1
fi

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
IMGSZ="${2:-448}"
BATCH="${3:-8}"
MULTISPECTRAL="${4:-0}"
EPOCHS=100

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }
[[ "${BATCH}" =~ ^[0-9]+$ && "${BATCH}" -ge 1 ]] || { echo "BATCH must be positive integer"; exit 1; }

TASK=""
PROJECT=""
COMMON_PARAMS=""
declare -a JOBS=()

submit_job() {
  local cfg="$1"
  local freeze="$2"
  local label
  local ts
  label="$(basename "${cfg}" .yaml)"
  ts="$(date +%m%d-%H%M%S)"
  local varlist="${COMMON_PARAMS},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${label},CONFIG_YAML=${cfg}"
  echo "→ ${label}  [FREEZE=${freeze}]"
  qsub -V -v "${varlist}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

case "${PROFILE}" in
  segment_no_p2)
    TASK="segment"
    IMGSZ="${2:-448}"
    BATCH="${3:-8}"
    PROJECT="backbone_sweep"
    COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,GAUSSIAN_BLUR_P=0.25,WANDB=true,PROJECT=${PROJECT}"
    JOBS=(
      "$(timm_backbone_sweep_config segment resnet50 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment resnext50_32x4d no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment resnest50d no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment seresnet50 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment skresnext50_32x4d no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment efficientnet_b1 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment efficientnet_b3 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment efficientnetv2_s no_p2 pafpn)|0"
      "$(yolo_segment_no_p2_pafpn_config yolo11x)|0"
      "$(yolo_segment_no_p2_pafpn_config yolo11n)|0"
      "$(yolo_segment_no_p2_pafpn_config yolo12x)|0"
      "$(yolo_segment_no_p2_pafpn_config yolo12n)|0"
      "$(timm_backbone_sweep_config segment convnext_small no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment convnext_base no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment convnextv2_base no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment regnety_040 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment regnety_080 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment densenet121 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment hrnet_w18 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment hrnet_w32 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment swin_tiny_patch4_window7_224 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment pvt_tiny no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment pvt_v2_b2 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment deit_small_p16_224 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment maxvit_small no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment mobilevit_s no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment coatnet_0 no_p2 pafpn)|0"
      "$(timm_backbone_sweep_config segment dinov3_vitb14 no_p2 pafpn)|1"
    )
    ;;
  segment_no_p2_stoc_drop)
    TASK="segment"
    IMGSZ="${2:-448}"
    BATCH="${3:-8}"
    PROJECT="backbone_sweep_stoc"
    COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,GAUSSIAN_BLUR_P=0.25,WANDB=true,PROJECT=${PROJECT}"
    # Preserves the previous script state: no active jobs were enabled.
    JOBS=()
    ;;
  obb_base)
    TASK="obb"
    IMGSZ="${2:-224}"
    BATCH="${3:-16}"
    PROJECT="backbone_sweep"
    COMMON_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0,WANDB=true,PROJECT=${PROJECT}"
    JOBS=(
      "$(yolo_obb_pafpn_config yolo11x)|0"
      "$(yolo_obb_pafpn_config yolo12x)|0"
    )
    ;;
  obb_stoc_drop)
    TASK="obb"
    IMGSZ="${2:-224}"
    BATCH="${3:-16}"
    PROJECT="backbone_sweep_stoc_obb"
    COMMON_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${BATCH},CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0,WANDB=true,PROJECT=${PROJECT}"
    JOBS=(
      "$(timm_backbone_sweep_config obb resnet50 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb resnext50_32x4d stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb resnest50d stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb seresnet50 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb skresnext50_32x4d stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb efficientnet_b1 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb efficientnet_b3 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb efficientnetv2_s stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb convnext_small stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb convnext_base stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb convnextv2_base stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb regnety_040 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb regnety_080 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb densenet121 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb hrnet_w18 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb hrnet_w32 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb swin_tiny_patch4_window7_224 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb pvt_tiny stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb pvt_v2_b2 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb deit_small_p16_224 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb maxvit_small stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb mobilevit_s stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb coatnet_0 stoc_drop pafpn)|0"
      "$(timm_backbone_sweep_config obb dinov3_vitb14 stoc_drop pafpn)|1"
    )
    ;;
  *)
    echo "[ERROR] Unsupported profile: ${PROFILE}" >&2
    exit 1
    ;;
esac

ACTIVE_COUNT=0
for row in "${JOBS[@]}"; do
  [[ -n "${row}" ]] || continue
  IFS='|' read -r cfg _freeze <<< "${row}"
  timm_require_file "${cfg}"
  ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
done

echo "Submitting ${ACTIVE_COUNT} jobs [PROFILE=${PROFILE}, TASK=${TASK}, IMGSZ=${IMGSZ}, BATCH=${BATCH}, MULTISPECTRAL=${MULTISPECTRAL}]"

if [[ "${ACTIVE_COUNT}" -eq 0 ]]; then
  echo "No jobs are active for profile ${PROFILE}."
  exit 0
fi

i=0
for row in "${JOBS[@]}"; do
  [[ -n "${row}" ]] || continue
  IFS='|' read -r cfg freeze <<< "${row}"
  i=$((i + 1))
  printf '[%02d/%02d] ' "${i}" "${ACTIVE_COUNT}"
  submit_job "${cfg}" "${freeze}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
