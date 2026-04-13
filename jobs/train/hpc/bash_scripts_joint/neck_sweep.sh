#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  echo "Usage: bash jobs/train/hpc/bash_scripts_joint/neck_sweep.sh <profile> [IMGSZ] [BATCH]"
  echo "Profiles: seg_non512, seg_512, seg_stoc_depth_512, obb_non512, obb_512"
  exit 1
fi

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
IMGSZ="${2:-448}"
BATCH="${3:-8}"
FREEZE="0"
COMMON_PARAMS=""
declare -a JOBS=()

submit_job() {
  local cfg="$1"
  local label
  local ts
  label="$(basename "${cfg}" .yaml)"
  ts="$(date +%m%d-%H%M%S)"
  local varlist="${COMMON_PARAMS},EXPERIMENT_MODE=${ts}_${label},FREEZE=${FREEZE},CONFIG_YAML=${cfg}"
  echo "→ ${label}"
  qsub -V -v "${varlist}" -N "${label}" "${PBS_SCRIPT}"
  sleep 1
}

case "${PROFILE}" in
  seg_non512)
    IMGSZ="${2:-224}"
    BATCH="${3:-16}"
    FREEZE="0"
    COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=0,BATCH=${BATCH},WANDB=true,PROJECT=neck_sweep_segment,CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,GAUSSIAN_BLUR_P=0.25"
    JOBS=(
      "$(timm_neck_sweep_config segment hrnet_w32 fpn_se base)"
      "$(timm_neck_sweep_config segment hrnet_w32 pafpn_cbam_sp base)"
      "$(timm_neck_sweep_config segment hrnet_w32 pafpn_se base)"
    )
    ;;
  seg_512)
    IMGSZ="${2:-448}"
    BATCH="${3:-8}"
    FREEZE="0"
    COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=0,BATCH=${BATCH},WANDB=true,PROJECT=neck_sweep_segment,CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,GAUSSIAN_BLUR_P=0.25"
    mapfile -t JOBS < <(find ultralytics/cfg/models/timm/segment/sweeps/neck/hrnet/hrnet_w32/base -maxdepth 1 -type f -name '*512c*-segment.yaml' | sort)
    ;;
  seg_stoc_depth_512)
    IMGSZ="${2:-448}"
    BATCH="${3:-8}"
    FREEZE="0"
    COMMON_PARAMS="TASK=segment,IMGSZ=${IMGSZ},CHECKPOINT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=0,BATCH=${BATCH},WANDB=true,PROJECT=neck_sweep_segment_stoc_depth,CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,GAUSSIAN_BLUR_P=0.25"
    mapfile -t JOBS < <(find ultralytics/cfg/models/timm/segment/sweeps/neck/hrnet/hrnet_w32/with_stoc_depth -maxdepth 1 -type f -name '*512c*-segment.yaml' | sort)
    ;;
  obb_non512)
    IMGSZ="${2:-224}"
    BATCH="${3:-16}"
    FREEZE="1"
    COMMON_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=0,BATCH=${BATCH},WANDB=true,PROJECT=neck_sweep,CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
    JOBS=(
      "$(timm_neck_sweep_config obb dinov3_7_12_17_22 fpn_carafe base)"
      "$(timm_neck_sweep_config obb dinov3_7_12_17_22 pafpn_carafe base)"
    )
    ;;
  obb_512)
    IMGSZ="${2:-224}"
    BATCH="${3:-16}"
    FREEZE="1"
    COMMON_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=0,BATCH=${BATCH},WANDB=true,PROJECT=neck_sweep,CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
    JOBS=(
      "$(timm_neck_sweep_config obb dinov3_7_12_17_22 fpn_carafe-seg_512c base)"
      "$(timm_neck_sweep_config obb dinov3_7_12_17_22 pafpn_carafe-seg_512c base)"
      "$(timm_neck_sweep_config obb dinov3_7_12_17_22 pafpn_se-seg_512c base)"
    )
    ;;
  *)
    echo "[ERROR] Unsupported profile: ${PROFILE}" >&2
    exit 1
    ;;
esac

ACTIVE_COUNT=0
for cfg in "${JOBS[@]}"; do
  [[ -n "${cfg}" ]] || continue
  timm_require_file "${cfg}"
  ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
done

echo "Submitting ${ACTIVE_COUNT} jobs [PROFILE=${PROFILE}, IMGSZ=${IMGSZ}, BATCH=${BATCH}]"

if [[ "${ACTIVE_COUNT}" -eq 0 ]]; then
  echo "No jobs are active for profile ${PROFILE}."
  exit 0
fi

i=0
for cfg in "${JOBS[@]}"; do
  [[ -n "${cfg}" ]] || continue
  i=$((i + 1))
  printf '[%02d/%02d] ' "${i}" "${ACTIVE_COUNT}"
  submit_job "${cfg}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
