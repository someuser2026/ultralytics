#!/usr/bin/env bash
set -euo pipefail

# Submit the fixed shoreline/land-water segment model checks requested for the
# Planet full c448 10075-single shoreline-band dataset.
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_seg/submit_shoreline_segment_models.sh [BATCH] [EPOCHS] [SEED] [DRY_RUN]
#
# Example:
#   bash jobs/train/hpc/bash_scripts_seg/submit_shoreline_segment_models.sh 8 5 0 1
#
# Optional environment overrides:
#   DATA_YAML=/path/to/data.yaml
#   PROJECT=null
#   DEVICE=0
#   WANDB=true

IMGSZ=448
BATCH="${1:-8}"
EPOCHS="${2:-5}"
SEED="${3:-0}"
DRY_RUN="${4:-0}"
DEVICE="${DEVICE:-0}"
WANDB="${WANDB:-true}"
PROJECT="${PROJECT:-null}"
PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

[[ "$BATCH" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "BATCH must be numeric"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || { echo "EPOCHS must be an integer"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be an integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }

if [[ -z "${DATA_YAML:-}" ]]; then
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH must be set unless DATA_YAML is provided" >&2
    exit 1
  fi
  DATA_YAML="${SCRATCH}/data_processed/Global/Annotated/variants/segment/planet_full_c448_ov35_kf20_10075-single_sh-lw-d-prx-cl-hz-sdw_seed0/data.yaml"
fi

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[ERROR] Missing config: $path" >&2; exit 1; }
}

append_var() {
  local key="$1"
  local value="$2"
  VARS+=("${key}=${value}")
}

submit_job() {
  local label="$1"
  local config_yaml="$2"
  local use_shoreline_prior_loss="$3"
  local use_land_water_prior_loss="$4"
  local use_shoreline_input="$5"
  local use_land_water_input="$6"
  local use_shoreline_aux_loss="$7"
  local shoreline_aux_warmup_epochs="$8"

  VARS=()
  append_var "TASK" "segment"
  append_var "IMGSZ" "$IMGSZ"
  append_var "CHECKPOINT" "null"
  append_var "TIME_FLOAT" "null"
  append_var "EPOCHS" "$EPOCHS"
  append_var "DEVICE" "$DEVICE"
  append_var "EXPERIMENT_MODE" "$label"
  append_var "OVERLAP" "35"
  append_var "KEEP_FRAC" "20"
  append_var "MULTISPECTRAL" "003"
  append_var "DATA_YAML" "$DATA_YAML"
  append_var "BATCH" "$BATCH"
  append_var "CONFIG_YAML" "$config_yaml"
  append_var "FREEZE" "0"
  append_var "SEED" "$SEED"
  append_var "WANDB" "$WANDB"
  append_var "PROJECT" "$PROJECT"
  append_var "USE_SHORELINE_PRIOR_LOSS" "$use_shoreline_prior_loss"
  append_var "USE_LAND_WATER_PRIOR_LOSS" "$use_land_water_prior_loss"
  append_var "USE_SHORELINE_INPUT" "$use_shoreline_input"
  append_var "USE_LAND_WATER_INPUT" "$use_land_water_input"
  append_var "USE_SHORELINE_AUX_LOSS" "$use_shoreline_aux_loss"
  if [[ "$shoreline_aux_warmup_epochs" != "null" ]]; then
    append_var "SHORELINE_AUX_WARMUP_EPOCHS" "$shoreline_aux_warmup_epochs"
  fi

  local varlist
  varlist="$(IFS=,; echo "${VARS[*]}")"
  local cmd=(qsub -V -v "$varlist" -N "$label" "$PBS_SCRIPT")

  echo "-> ${label} | cfg=${config_yaml}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '   DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
    sleep 1
  fi
}

MODELS=(
  "yolo12n_seg_shore_lw_loss|ultralytics/cfg/models/12/yolo12-seg.yaml|true|true|false|false|false|null"
  "yolo12n_seg_shore_lw_input|ultralytics/cfg/models/12/yolo12-seg.yaml|false|false|true|true|false|null"
  "yolo12n_seg_shore_aux_head|ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml|false|false|false|false|true|2"
  "yolo26n_seg_normal|ultralytics/cfg/models/26/yolo26-seg.yaml|false|false|false|false|false|null"
  "yolo26n_seg_shore_lw_input|ultralytics/cfg/models/26/yolo26-seg.yaml|false|false|true|true|false|null"
  "mamba_hrnet_seg_shore_lw_loss|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|true|true|false|false|false|null"
  "mamba_hrnet_seg_shore_lw_input|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|false|false|true|true|false|null"
  "mamba_hrnet_yolo26_seg_normal|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml|false|false|false|false|false|null"
  "mamba_hrnet_yolo26_seg_shore_lw_input|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml|false|false|true|true|false|null"
  "mamba_hrnet_cascade_mask_rcnn_normal|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-cascade-mask-rcnn.yaml|false|false|false|false|false|null"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r _label cfg _shore_prior _lw_prior _shore_input _lw_input _shore_aux _warmup <<< "$entry"
  require_file "$cfg"
done

echo "Submitting ${#MODELS[@]} shoreline segment jobs"
echo "DATA_YAML=${DATA_YAML}"
echo "IMGSZ=${IMGSZ}, EPOCHS=${EPOCHS}, BATCH=${BATCH}, DEVICE=${DEVICE}, SEED=${SEED}, PROJECT=${PROJECT}, DRY_RUN=${DRY_RUN}"

count=0
for entry in "${MODELS[@]}"; do
  count=$((count + 1))
  IFS='|' read -r label cfg shore_prior lw_prior shore_input lw_input shore_aux warmup <<< "$entry"
  printf '[%02d/%02d] ' "$count" "${#MODELS[@]}"
  submit_job "$label" "$cfg" "$shore_prior" "$lw_prior" "$shore_input" "$lw_input" "$shore_aux" "$warmup"
done

echo "Done. Monitor with: qstat -u \"$USER\""
