#!/usr/bin/env bash
# Submit joint PointRend segmentation training for the canonical single-class model configs.
# Usage: bash jobs/train/hpc/bash_scripts_seg/submit_pointrend_joint_segment_models.sh [batch] [epochs] [seed] [dry_run] [workers]

set -euo pipefail

for deprecated_var in MODEL ARCH PRETRAINED MODEL_YAML; do
    if [[ -n "${!deprecated_var:-}" ]]; then
        echo "Error: ${deprecated_var} is not supported by this submitter." >&2
        exit 2
    fi
done

IMGSZ="${IMGSZ:-448}"
BATCH="${1:-${BATCH:-4}}"
EPOCHS="${2:-${EPOCHS:-100}}"
SEED="${3:-${SEED:-0}}"
DRY_RUN="${4:-${DRY_RUN:-0}}"
WORKERS="${5:-${WORKERS:-1}}"
DEVICE="${DEVICE:-0}"
WANDB="${WANDB:-true}"
PROJECT="${PROJECT:-pointrend_joint_segment}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

require_file() {
    [[ -f "$1" ]] || { echo "Error: required file not found: $1" >&2; exit 2; }
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

append_var() {
    VARS+=("$1=$2")
}

is_uint "$IMGSZ" && (( IMGSZ >= 32 )) || { echo "Error: IMGSZ must be an integer >= 32." >&2; exit 2; }
is_uint "$BATCH" && (( BATCH >= 1 )) || { echo "Error: batch must be an integer >= 1." >&2; exit 2; }
is_uint "$EPOCHS" && (( EPOCHS >= 1 )) || { echo "Error: epochs must be an integer >= 1." >&2; exit 2; }
is_uint "$SEED" || { echo "Error: seed must be a non-negative integer." >&2; exit 2; }
is_uint "$WORKERS" || { echo "Error: workers must be a non-negative integer." >&2; exit 2; }
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || { echo "Error: dry_run must be 0 or 1." >&2; exit 2; }

require_file "$PBS_SCRIPT"

submit_job() {
    local label="$1"
    local config_yaml="$2"
    VARS=()

    require_file "$config_yaml"
    append_var TASK "segment"
    append_var IMGSZ "$IMGSZ"
    append_var CHECKPOINT "null"
    append_var TIME_FLOAT "null"
    append_var EPOCHS "$EPOCHS"
    append_var DEVICE "$DEVICE"
    append_var EXPERIMENT_MODE "${RUN_TAG}_${label}"
    append_var OVERLAP "35"
    append_var KEEP_FRAC "20"
    append_var MULTISPECTRAL "21"
    append_var BATCH "$BATCH"
    append_var WORKERS "$WORKERS"
    append_var CONFIG_YAML "$config_yaml"
    append_var FREEZE "0"
    append_var SEED "$SEED"
    append_var WANDB "$WANDB"
    append_var PROJECT "$PROJECT"
    append_var POINTREND_MODE "joint"
    append_var USE_SOFT_IGNORE "false"
    append_var CLAHE_P "0.0"
    append_var MOTION_BLUR_P "0.0"
    append_var MULTI_SPEC_NOISE_P "0.0"
    append_var UNSHARP_P "0.0"
    append_var GAUSSIAN_BLUR_P "0.0"
    append_var SDICE "1"

    local varlist
    varlist="$(IFS=,; echo "${VARS[*]}")"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY RUN: qsub -V -v ${varlist} -N ${label} ${PBS_SCRIPT}"
    else
        qsub -V -v "$varlist" -N "$label" "$PBS_SCRIPT"
    fi
}

declare -a jobs=(
    "pr_mhr|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg-pointrend.yaml"
    "pr_y12x|ultralytics/cfg/models/12/yolo12x-seg-pointrend.yaml"
    "pr_hr32|ultralytics/cfg/models/timm/segment/final/panet_adaptive/hrnet/hrnet_w32/four_scale/1cls/hrnet_w32-panet_adaptive-segment-pointrend.yaml"
    "pr_y11x|ultralytics/cfg/models/11/yolo11x-seg-pointrend.yaml"
    "pr_y26|ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml"
)

echo "Submitting ${#jobs[@]} PointRend joint-training jobs (imgsz=${IMGSZ}, batch=${BATCH}, epochs=${EPOCHS})."
for job in "${jobs[@]}"; do
    IFS='|' read -r label config_yaml <<< "$job"
    submit_job "$label" "$config_yaml"
done
