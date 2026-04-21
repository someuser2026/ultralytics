#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh [MODEL_OR_CONFIG] [IMGSZ] [BATCH] [MULTISPECTRAL_OPT] [DRY_RUN]

Examples:
  bash jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh
  bash jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh mamba-yolo-l-obb 224 16 pn10075s 1
  bash jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh mamba-hrnet-obb 448 8 pn10075s 0
  bash jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml 256 8 0 0

Supported aliases:
  mamba-yolo-l-obb
  mamba-yolo-obb
  Mamba-YOLO-L-obb-demo
  mamba-hrnet-obb
  mamba_hrnet_obb

Optional environment overrides:
  EPOCHS=100
  TIME_FLOAT=null
  DEVICE=0
  OVERLAP=35
  KEEP_FRAC=20
  FREEZE=0
  SEED=0
  CHECKPOINT=null
  PROJECT=mamba_yolo_obb
  WANDB=true
  JOB_LABEL=<config-stem>_<MS_TAG>
  EXPERIMENT_MODE=<timestamped default>
  ANGLE_MODE=le90

Common augmentation overrides:
  CLAHE_P=0.25
  RAND_GAMMA_P=
  UNSHARP_P=0.0
  EDGEBOOST_P=
  HOMOMORPHIC_P=
  GAUSSIAN_BLUR_P=0.0
  MOTION_BLUR_P=0.0
  ADDITIVE_NOISE_P=
  MULTI_SPEC_NOISE_P=0.50
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_OR_CONFIG="${1:-mamba-yolo-l-obb}"
IMGSZ="${2:-224}"
BATCH="${3:-16}"
MULTISPECTRAL_OPT="${4:-21}"
DRY_RUN="${5:-${DRY_RUN:-0}}"

[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }
[[ "$BATCH" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "BATCH must be numeric"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

resolve_ms() {
  local opt="$1"
  case "$opt" in
    0|rgb|rgb100)
      MS_CODE="0";   MS_TAG="RGB100";     MS_DESC="Planet RGB (100 only)" ;;
    01|rgbma|rgb-max|rgbmax|rgbmaxarea)
      MS_CODE="01";  MS_TAG="RGBMA";      MS_DESC="Planet RGB (maxarea5000)" ;;
    001|rgb10075m|rgb-10075m|10075-multi)
      MS_CODE="001"; MS_TAG="RGB10075M";  MS_DESC="Planet RGB (100+75 multi-class)" ;;
    002|rgball|rgb-all|all)
      MS_CODE="002"; MS_TAG="RGBALL";     MS_DESC="Planet RGB (100+75+50+30 all classes)" ;;
    003|rgb10075s|rgb-10075s|10075-single)
      MS_CODE="003"; MS_TAG="RGB10075S";  MS_DESC="Planet RGB (100+75 collapsed)" ;;
    004|rgbsoft|all-soft)
      MS_CODE="004"; MS_TAG="RGBSOFT";    MS_DESC="Planet RGB soft-label variant" ;;
    1|nir|msnir)
      MS_CODE="1";   MS_TAG="NIR";        MS_DESC="Planet RGB+NIR (analytic_udm2)" ;;
    11|nirma|nirmax|nir-max|nirmaxarea)
      MS_CODE="11";  MS_TAG="NIRMA";      MS_DESC="Planet RGB+NIR (maxarea5000)" ;;
    2|pn|pn100|planet_nearmap)
      MS_CODE="2";   MS_TAG="PN100";      MS_DESC="Planet+Nearmap (100 only)" ;;
    21|pn10075s|pn-10075s)
      MS_CODE="21";  MS_TAG="PN10075S";   MS_DESC="Planet+Nearmap (100+75 collapsed)" ;;
    3|nm|nm100|nearmap)
      MS_CODE="3";   MS_TAG="NM100";      MS_DESC="Nearmap only (100 only)" ;;
    *)
      echo "[ERROR] Unknown MULTISPECTRAL_OPT='$opt'"
      echo "Valid options: 0, 01, 001, 002, 003, 004, 1, 11, 2, 21, 3"
      exit 1
      ;;
  esac
}

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

to_repo_relative() {
  local path="$1"
  case "$path" in
    "${REPO_ROOT}/"*)
      printf '%s\n' "${path#${REPO_ROOT}/}"
      ;;
    *)
      printf '%s\n' "$path"
      ;;
  esac
}

resolve_config() {
  local model_or_config="$1"
  case "$model_or_config" in
    mamba-yolo-l-obb|mamba-yolo-l-obb.yaml|mamba-yolo-obb|mamba-yolo-obb.yaml|Mamba-YOLO-L-obb-demo|Mamba-YOLO-L-obb-demo.yaml)
      printf 'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml\n'
      return 0
      ;;
    mamba-hrnet-obb|mamba-hrnet-obb.yaml|mamba_hrnet_obb|mamba_hrnet_obb.yaml)
      printf 'ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb.yaml\n'
      return 0
      ;;
  esac

  local resolved
  resolved="$(resolve_path "$model_or_config")"
  [[ -f "$resolved" ]] || {
    echo "Could not resolve MODEL_OR_CONFIG to an existing file: $model_or_config" >&2
    return 1
  }
  to_repo_relative "$resolved"
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

resolve_ms "$MULTISPECTRAL_OPT"

EPOCHS="${EPOCHS:-100}"
TIME_FLOAT="${TIME_FLOAT:-null}"
DEVICE="${DEVICE:-0}"
OVERLAP="${OVERLAP:-35}"
KEEP_FRAC="${KEEP_FRAC:-20}"
FREEZE="${FREEZE:-0}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
PROJECT="${PROJECT:-mamba_yolo_obb}"
WANDB="${WANDB:-true}"
ANGLE_MODE="${ANGLE_MODE:-le90}"

CLAHE_P="${CLAHE_P:-0.25}"
RAND_GAMMA_P="${RAND_GAMMA_P:-}"
UNSHARP_P="${UNSHARP_P:-0.0}"
EDGEBOOST_P="${EDGEBOOST_P:-}"
HOMOMORPHIC_P="${HOMOMORPHIC_P:-}"
GAUSSIAN_BLUR_P="${GAUSSIAN_BLUR_P:-0.0}"
MOTION_BLUR_P="${MOTION_BLUR_P:-0.0}"
ADDITIVE_NOISE_P="${ADDITIVE_NOISE_P:-}"
MULTI_SPEC_NOISE_P="${MULTI_SPEC_NOISE_P:-0.50}"

CONFIG_YAML="$(resolve_config "$MODEL_OR_CONFIG")"
CONFIG_BASENAME="$(basename "$CONFIG_YAML")"
CONFIG_STEM="${CONFIG_BASENAME%.yaml}"
SAFE_CONFIG_STEM="${CONFIG_STEM//[^A-Za-z0-9_.-]/_}"

if [[ "$EPOCHS" == "null" && "$TIME_FLOAT" == "null" ]]; then
  echo "Either EPOCHS or TIME_FLOAT must be set"
  exit 1
fi

TS="$(date +%m%d-%H%M%S)"
JOB_LABEL="${JOB_LABEL:-${SAFE_CONFIG_STEM}_${MS_TAG}}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-${TS}_${SAFE_CONFIG_STEM}_obb_${MS_TAG}}"

VARS=()
append_var "TASK" "obb"
append_var "IMGSZ" "$IMGSZ"
append_var "CHECKPOINT" "$CHECKPOINT"
append_var "TIME_FLOAT" "$TIME_FLOAT"
append_var "EPOCHS" "$EPOCHS"
append_var "DEVICE" "$DEVICE"
append_var "EXPERIMENT_MODE" "$EXPERIMENT_MODE"
append_var "OVERLAP" "$OVERLAP"
append_var "KEEP_FRAC" "$KEEP_FRAC"
append_var "MULTISPECTRAL" "$MS_CODE"
append_var "BATCH" "$BATCH"
append_var "CONFIG_YAML" "$CONFIG_YAML"
append_var "FREEZE" "$FREEZE"
append_var "SEED" "$SEED"
append_var "WANDB" "$WANDB"
append_var "PROJECT" "$PROJECT"

append_if_set "ANGLE_MODE" "$ANGLE_MODE"
append_if_set "CLAHE_P" "$CLAHE_P"
append_if_set "RAND_GAMMA_P" "$RAND_GAMMA_P"
append_if_set "UNSHARP_P" "$UNSHARP_P"
append_if_set "EDGEBOOST_P" "$EDGEBOOST_P"
append_if_set "HOMOMORPHIC_P" "$HOMOMORPHIC_P"
append_if_set "GAUSSIAN_BLUR_P" "$GAUSSIAN_BLUR_P"
append_if_set "MOTION_BLUR_P" "$MOTION_BLUR_P"
append_if_set "ADDITIVE_NOISE_P" "$ADDITIVE_NOISE_P"
append_if_set "MULTI_SPEC_NOISE_P" "$MULTI_SPEC_NOISE_P"

VARLIST="$(IFS=,; echo "${VARS[*]}")"
CMD=(qsub -V -v "$VARLIST" -N "$JOB_LABEL" "$PBS_SCRIPT")

echo "============================================================"
echo "Submitting Mamba-YOLO OBB job"
echo "  Config: ${CONFIG_YAML}"
echo "  Image size: ${IMGSZ}"
echo "  Batch: ${BATCH}"
echo "  Data: ${MS_DESC} (${MS_TAG})"
echo "  Project: ${PROJECT}"
echo "  Experiment mode: ${EXPERIMENT_MODE}"
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
