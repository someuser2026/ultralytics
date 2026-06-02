#!/usr/bin/env bash
set -euo pipefail

for deprecated_var in USE_SHORELINE_INPUT USE_LAND_WATER_INPUT; do
  if [[ -n "${!deprecated_var:-}" ]]; then
    echo "${deprecated_var} is no longer supported. Configure selected model input bands with input_bands in data.yaml." >&2
    exit 1
  fi
done

# Submit the fixed shoreline segment model set across dataset YAMLs that select
# different model input bands from the same 10-band TIFF files.
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_seg/submit_input_band_segment_models.sh [BATCH] [EPOCHS] [SEED] [DRY_RUN] [WORKERS]
#
# Expected dataset YAMLs:
#   data.yaml        -> input_bands: [1, 2, 3]
#   data_sh_lw.yaml  -> input_bands: [1, 2, 3, 4, 5]
#   data_all.yaml    -> no input_bands key, model receives all raw channels
#
# Optional environment overrides:
#   DATA_ROOT=/path/to/dataset/root
#   DATA_RGB_YAML=/path/to/data.yaml
#   DATA_SH_LW_YAML=/path/to/data_sh_lw.yaml
#   DATA_ALL_YAML=/path/to/data_all.yaml
#   PROJECT=input_band_segment_models
#   DEVICE=0
#   WANDB=true
#   WORKERS=1

IMGSZ=448
BATCH="${1:-${BATCH:-4}}"
EPOCHS="${2:-${EPOCHS:-100}}"
SEED="${3:-${SEED:-0}}"
DRY_RUN="${4:-${DRY_RUN:-0}}"
WORKERS="${5:-${WORKERS:-1}}"
DEVICE="${DEVICE:-0}"
WANDB="${WANDB:-true}"
PROJECT="${PROJECT:-input_band_segment_models}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
USE_SOFT_IGNORE="${USE_SOFT_IGNORE:-}"
SDICE="${SDICE:-1}"
SBCE="${SBCE:-}"
SLOVHN="${SLOVHN:-}"
CLAHE_P="${CLAHE_P:-0.0}"
RAND_GAMMA_P="${RAND_GAMMA_P:-}"
UNSHARP_P="${UNSHARP_P:-0.0}"
EDGEBOOST_P="${EDGEBOOST_P:-}"
HOMOMORPHIC_P="${HOMOMORPHIC_P:-}"
GAUSSIAN_BLUR_P="${GAUSSIAN_BLUR_P:-0.0}"
MOTION_BLUR_P="${MOTION_BLUR_P:-0.0}"
ADDITIVE_NOISE_P="${ADDITIVE_NOISE_P:-}"
MULTI_SPEC_NOISE_P="${MULTI_SPEC_NOISE_P:-0.0}"
MOSAIC="${MOSAIC:-0.0}"
MIXUP="${MIXUP:-0.0}"
COPY_PASTE="${COPY_PASTE:-0.0}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-0}"
SEGMENT_PRIOR_TOPK="${SEGMENT_PRIOR_TOPK:-}"
PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

[[ "$BATCH" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "BATCH must be numeric"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || { echo "EPOCHS must be an integer"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be an integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ "$WORKERS" =~ ^[0-9]+$ ]] || { echo "WORKERS must be a non-negative integer"; exit 1; }

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -n "${DATA_RGB_YAML:-}" && -n "${DATA_SH_LW_YAML:-}" && -n "${DATA_ALL_YAML:-}" ]]; then
    :
  elif [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH must be set unless DATA_ROOT or all DATA_*_YAML paths are provided" >&2
    exit 1
  else
    DATA_ROOT="${SCRATCH}/data_processed/Global/Annotated/variants/segment/planet_full_c448_ov35_kf20_10075-single_sh-lw-d-prx-cl-hz-sdw_seed0"
  fi
fi

DATA_RGB_YAML="${DATA_RGB_YAML:-${DATA_ROOT}/data.yaml}"
DATA_SH_LW_YAML="${DATA_SH_LW_YAML:-${DATA_ROOT}/data_sh_lw.yaml}"
DATA_ALL_YAML="${DATA_ALL_YAML:-${DATA_ROOT}/data_all.yaml}"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[ERROR] Missing config: $path" >&2; exit 1; }
}

append_var() {
  local key="$1"
  local value="$2"
  VARS+=("${key}=${value}")
}

append_if_set() {
  local key="$1"
  local value="${2:-}"
  if [[ -n "$value" ]]; then
    VARS+=("${key}=${value}")
  fi
}

submit_job() {
  local data_tag="$1"
  local data_yaml="$2"
  local model_label="$3"
  local config_yaml="$4"
  local use_shoreline_prior_loss="$5"
  local use_land_water_prior_loss="$6"
  local use_shoreline_aux_loss="$7"
  local shoreline_aux_warmup_epochs="$8"
  local label="${model_label}_${data_tag}"

  VARS=()
  append_var "TASK" "segment"
  append_var "IMGSZ" "$IMGSZ"
  append_var "CHECKPOINT" "null"
  append_var "TIME_FLOAT" "null"
  append_var "EPOCHS" "$EPOCHS"
  append_var "DEVICE" "$DEVICE"
  append_var "EXPERIMENT_MODE" "${RUN_TAG}_${label}"
  append_var "OVERLAP" "35"
  append_var "KEEP_FRAC" "20"
  append_var "MULTISPECTRAL" "003"
  append_var "DATA_YAML" "$data_yaml"
  append_var "BATCH" "$BATCH"
  append_var "WORKERS" "$WORKERS"
  append_var "CONFIG_YAML" "$config_yaml"
  append_var "FREEZE" "0"
  append_var "SEED" "$SEED"
  append_var "WANDB" "$WANDB"
  append_var "PROJECT" "$PROJECT"
  append_if_set "USE_SOFT_IGNORE" "$USE_SOFT_IGNORE"
  append_if_set "SDICE" "$SDICE"
  append_if_set "SBCE" "$SBCE"
  append_if_set "SLOVHN" "$SLOVHN"
  append_var "USE_SHORELINE_PRIOR_LOSS" "$use_shoreline_prior_loss"
  append_var "USE_LAND_WATER_PRIOR_LOSS" "$use_land_water_prior_loss"
  append_var "USE_SHORELINE_AUX_LOSS" "$use_shoreline_aux_loss"
  append_if_set "SEGMENT_PRIOR_TOPK" "$SEGMENT_PRIOR_TOPK"
  if [[ "$shoreline_aux_warmup_epochs" != "null" ]]; then
    append_var "SHORELINE_AUX_WARMUP_EPOCHS" "$shoreline_aux_warmup_epochs"
  fi
  append_if_set "CLAHE_P" "$CLAHE_P"
  append_if_set "RAND_GAMMA_P" "$RAND_GAMMA_P"
  append_if_set "UNSHARP_P" "$UNSHARP_P"
  append_if_set "EDGEBOOST_P" "$EDGEBOOST_P"
  append_if_set "HOMOMORPHIC_P" "$HOMOMORPHIC_P"
  append_if_set "GAUSSIAN_BLUR_P" "$GAUSSIAN_BLUR_P"
  append_if_set "MOTION_BLUR_P" "$MOTION_BLUR_P"
  append_if_set "ADDITIVE_NOISE_P" "$ADDITIVE_NOISE_P"
  append_if_set "MULTI_SPEC_NOISE_P" "$MULTI_SPEC_NOISE_P"
  append_if_set "MOSAIC" "$MOSAIC"
  append_if_set "MIXUP" "$MIXUP"
  append_if_set "COPY_PASTE" "$COPY_PASTE"
  append_if_set "CLOSE_MOSAIC" "$CLOSE_MOSAIC"

  local varlist
  varlist="$(IFS=,; echo "${VARS[*]}")"
  local cmd=(qsub -V -v "$varlist" -N "$label" "$PBS_SCRIPT")

  echo "-> ${label} | data=${data_yaml} | cfg=${config_yaml}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '   DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
    sleep 1
  fi
}

DATASETS=(
  "rgb|${DATA_RGB_YAML}"
  "sh_lw|${DATA_SH_LW_YAML}"
  "all|${DATA_ALL_YAML}"
)

MODELS=(
  "yolo12n_seg_shore_lw_loss|ultralytics/cfg/models/12/yolo12-seg.yaml|true|true|false|null"
  "yolo12n_seg_shore_lw_input|ultralytics/cfg/models/12/yolo12-seg.yaml|false|false|false|null"
  "yolo12n_seg_shore_aux_head|ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml|false|false|true|2"
  "yolo26n_seg_normal|ultralytics/cfg/models/26/yolo26-seg.yaml|false|false|false|null"
  "yolo26n_seg_shore_lw_input|ultralytics/cfg/models/26/yolo26-seg.yaml|false|false|false|null"
  "mamba_hrnet_seg_shore_lw_loss|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|true|true|false|null"
  "mamba_hrnet_seg_shore_lw_input|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|false|false|false|null"
  "mamba_hrnet_yolo26_seg_normal|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml|false|false|false|null"
  "mamba_hrnet_yolo26_seg_shore_lw_input|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml|false|false|false|null"
  "mamba_hrnet_cascade_mask_rcnn_normal|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-cascade-mask-rcnn.yaml|false|false|false|null"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r _label cfg _shore_prior _lw_prior _shore_aux _warmup <<< "$entry"
  require_file "$cfg"
done

echo "Submitting $((${#DATASETS[@]} * ${#MODELS[@]})) input-band segment jobs"
echo "DATA_RGB_YAML=${DATA_RGB_YAML}"
echo "DATA_SH_LW_YAML=${DATA_SH_LW_YAML}"
echo "DATA_ALL_YAML=${DATA_ALL_YAML}"
echo "IMGSZ=${IMGSZ}, EPOCHS=${EPOCHS}, BATCH=${BATCH}, WORKERS=${WORKERS}, DEVICE=${DEVICE}, SEED=${SEED}, PROJECT=${PROJECT}, RUN_TAG=${RUN_TAG}, DRY_RUN=${DRY_RUN}"
echo "Augmentations: CLAHE_P=${CLAHE_P}, UNSHARP_P=${UNSHARP_P}, GAUSSIAN_BLUR_P=${GAUSSIAN_BLUR_P}, MOTION_BLUR_P=${MOTION_BLUR_P}, MULTI_SPEC_NOISE_P=${MULTI_SPEC_NOISE_P}, MOSAIC=${MOSAIC}, MIXUP=${MIXUP}, COPY_PASTE=${COPY_PASTE}, CLOSE_MOSAIC=${CLOSE_MOSAIC}"

count=0
total=$((${#DATASETS[@]} * ${#MODELS[@]}))
for dataset_entry in "${DATASETS[@]}"; do
  IFS='|' read -r data_tag data_yaml <<< "$dataset_entry"
  for model_entry in "${MODELS[@]}"; do
    count=$((count + 1))
    IFS='|' read -r label cfg shore_prior lw_prior shore_aux warmup <<< "$model_entry"
    printf '[%02d/%02d] ' "$count" "$total"
    submit_job "$data_tag" "$data_yaml" "$label" "$cfg" "$shore_prior" "$lw_prior" "$shore_aux" "$warmup"
  done
done

echo "Done. Monitor with: qstat -u \"$USER\""
