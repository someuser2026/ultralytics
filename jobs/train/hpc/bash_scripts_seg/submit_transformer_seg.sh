#!/usr/bin/env bash

set -euo pipefail

for deprecated_var in USE_SHORELINE_INPUT USE_LAND_WATER_INPUT; do
  if [[ -n "${!deprecated_var:-}" ]]; then
    echo "${deprecated_var} is no longer supported. Configure selected model input bands with input_bands in data.yaml." >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh [MODEL_OR_CONFIG|all] [BATCH] [DRY_RUN]

Examples:
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh all 4 1
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh 4 1
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh mask2former-yolo12-seg 4 1
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh mask2former-swin-timm-seg 4 0
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh rtdetr-l-instance-seg 4 0
  bash jobs/train/hpc/bash_scripts_seg/submit_transformer_seg.sh ultralytics/cfg/models/transformer/mask2former-mamba-hrnet-seg.yaml 4 0

Supported aliases:
  all
  mask2former-mamba-hrnet-seg
  mask2former_mamba_hrnet_seg
  mask2former-swin-timm-seg
  mask2former_swin_timm_seg
  mask2former-yolo12-seg
  mask2former_yolo12_seg
  rtdetr-l-instance-seg
  rtdetr_l_instance_seg

Optional environment overrides:
  EPOCHS=100
  TIME_FLOAT=null
  DEVICE=0
  FREEZE=0
  SEED=0
  CHECKPOINT=null
  PROJECT=transformer_segment
  WANDB=true
  JOB_LABEL=<config-stem>_<MS_TAG>
  EXPERIMENT_MODE=<timestamped default>
  USE_SOFT_IGNORE=true
  SDICE=1
  SBCE=1
  SLOVHN=0.0

Common augmentation overrides:
  CLAHE_P=0.0
  RAND_GAMMA_P=
  UNSHARP_P=0.50
  EDGEBOOST_P=
  HOMOMORPHIC_P=
  GAUSSIAN_BLUR_P=0.25
  MOTION_BLUR_P=0.50
  ADDITIVE_NOISE_P=
  MULTI_SPEC_NOISE_P=0.10
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${1:-}" || "${1:-}" == "all" ]]; then
  RUN_ALL=1
  MODEL_OR_CONFIG="all"
  BATCH="${2:-${BATCH:-4}}"
  DRY_RUN="${3:-${DRY_RUN:-0}}"
elif [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  RUN_ALL=1
  MODEL_OR_CONFIG="all"
  BATCH="$1"
  DRY_RUN="${2:-${DRY_RUN:-0}}"
else
  RUN_ALL=0
  MODEL_OR_CONFIG="$1"
  BATCH="${2:-${BATCH:-4}}"
  DRY_RUN="${3:-${DRY_RUN:-0}}"
fi

[[ "$BATCH" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "BATCH must be numeric"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
IMGSZ="448"
OVERLAP="35"
KEEP_FRAC="20"
MS_CODE="21"
MS_TAG="PN10075S"
MS_DESC="Planet+Nearmap (100+75 collapsed)"

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
    mask2former-mamba-hrnet-seg|mask2former-mamba-hrnet-seg.yaml|mask2former_mamba_hrnet_seg|mask2former_mamba_hrnet_seg.yaml)
      printf 'ultralytics/cfg/models/transformer/mask2former-mamba-hrnet-seg.yaml\n'
      return 0
      ;;
    mask2former-swin-timm-seg|mask2former-swin-timm-seg.yaml|mask2former_swin_timm_seg|mask2former_swin_timm_seg.yaml)
      printf 'ultralytics/cfg/models/transformer/mask2former-swin-timm-seg.yaml\n'
      return 0
      ;;
    mask2former-yolo12-seg|mask2former-yolo12-seg.yaml|mask2former_yolo12_seg|mask2former_yolo12_seg.yaml)
      printf 'ultralytics/cfg/models/transformer/mask2former-yolo12-seg.yaml\n'
      return 0
      ;;
    rtdetr-l-instance-seg|rtdetr-l-instance-seg.yaml|rtdetr_l_instance_seg|rtdetr_l_instance_seg.yaml)
      printf 'ultralytics/cfg/models/transformer/rtdetr-l-instance-seg.yaml\n'
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

all_transformer_configs() {
  local cfg
  while IFS= read -r cfg; do
    to_repo_relative "$cfg"
  done < <(find "${REPO_ROOT}/ultralytics/cfg/models/transformer" -maxdepth 1 -type f -name '*.yaml' | sort)
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

EPOCHS="${EPOCHS:-100}"
TIME_FLOAT="${TIME_FLOAT:-null}"
DEVICE="${DEVICE:-0}"
FREEZE="${FREEZE:-0}"
SEED="${SEED:-0}"
CHECKPOINT="${CHECKPOINT:-null}"
PROJECT="${PROJECT:-transformer_segment}"
WANDB="${WANDB:-true}"
USE_SOFT_IGNORE="${USE_SOFT_IGNORE:-}"
SDICE="${SDICE:-1}"
SBCE="${SBCE:-}"
SLOVHN="${SLOVHN:-}"
SEGMENT_PRIOR_TOPK="${SEGMENT_PRIOR_TOPK:-}"

CLAHE_P="${CLAHE_P:-0.0}"
RAND_GAMMA_P="${RAND_GAMMA_P:-}"
UNSHARP_P="${UNSHARP_P:-0.50}"
EDGEBOOST_P="${EDGEBOOST_P:-}"
HOMOMORPHIC_P="${HOMOMORPHIC_P:-}"
GAUSSIAN_BLUR_P="${GAUSSIAN_BLUR_P:-0.25}"
MOTION_BLUR_P="${MOTION_BLUR_P:-0.50}"
ADDITIVE_NOISE_P="${ADDITIVE_NOISE_P:-}"
MULTI_SPEC_NOISE_P="${MULTI_SPEC_NOISE_P:-0.10}"

if [[ "$EPOCHS" == "null" && "$TIME_FLOAT" == "null" ]]; then
  echo "Either EPOCHS or TIME_FLOAT must be set"
  exit 1
fi

TS="$(date +%m%d-%H%M%S)"

submit_config() {
  local config_yaml="$1"
  local config_basename config_stem safe_config_stem job_label experiment_mode varlist

  config_basename="$(basename "$config_yaml")"
  config_stem="${config_basename%.yaml}"
  safe_config_stem="${config_stem//[^A-Za-z0-9_.-]/_}"

  if [[ -n "${JOB_LABEL:-}" ]]; then
    if [[ "$RUN_ALL" == "1" ]]; then
      job_label="${JOB_LABEL}_${safe_config_stem}"
    else
      job_label="$JOB_LABEL"
    fi
  else
    job_label="${safe_config_stem}_${MS_TAG}"
  fi

  if [[ -n "${EXPERIMENT_MODE:-}" ]]; then
    if [[ "$RUN_ALL" == "1" ]]; then
      experiment_mode="${EXPERIMENT_MODE}_${safe_config_stem}"
    else
      experiment_mode="$EXPERIMENT_MODE"
    fi
  else
    experiment_mode="${TS}_${safe_config_stem}_segment_${MS_TAG}"
  fi

  VARS=()
  append_var "TASK" "segment"
  append_var "IMGSZ" "$IMGSZ"
  append_var "CHECKPOINT" "$CHECKPOINT"
  append_var "TIME_FLOAT" "$TIME_FLOAT"
  append_var "EPOCHS" "$EPOCHS"
  append_var "DEVICE" "$DEVICE"
  append_var "EXPERIMENT_MODE" "$experiment_mode"
  append_var "OVERLAP" "$OVERLAP"
  append_var "KEEP_FRAC" "$KEEP_FRAC"
  append_var "MULTISPECTRAL" "$MS_CODE"
  append_var "BATCH" "$BATCH"
  append_var "CONFIG_YAML" "$config_yaml"
  append_var "FREEZE" "$FREEZE"
  append_var "SEED" "$SEED"
  append_var "WANDB" "$WANDB"
  append_var "PROJECT" "$PROJECT"

  append_if_set "USE_SOFT_IGNORE" "$USE_SOFT_IGNORE"
  append_if_set "SDICE" "$SDICE"
  append_if_set "SBCE" "$SBCE"
  append_if_set "SLOVHN" "$SLOVHN"
  append_if_set "SEGMENT_PRIOR_TOPK" "$SEGMENT_PRIOR_TOPK"

  append_if_set "CLAHE_P" "$CLAHE_P"
  append_if_set "RAND_GAMMA_P" "$RAND_GAMMA_P"
  append_if_set "UNSHARP_P" "$UNSHARP_P"
  append_if_set "EDGEBOOST_P" "$EDGEBOOST_P"
  append_if_set "HOMOMORPHIC_P" "$HOMOMORPHIC_P"
  append_if_set "GAUSSIAN_BLUR_P" "$GAUSSIAN_BLUR_P"
  append_if_set "MOTION_BLUR_P" "$MOTION_BLUR_P"
  append_if_set "ADDITIVE_NOISE_P" "$ADDITIVE_NOISE_P"
  append_if_set "MULTI_SPEC_NOISE_P" "$MULTI_SPEC_NOISE_P"

  varlist="$(IFS=,; echo "${VARS[*]}")"
  CMD=(qsub -V -v "$varlist" -N "$job_label" "$PBS_SCRIPT")

  echo "============================================================"
  echo "Submitting transformer segment job"
  echo "  Config: ${config_yaml}"
  echo "  Image size: ${IMGSZ}"
  echo "  Batch: ${BATCH}"
  echo "  Data: ${MS_DESC} (${MS_TAG})"
  echo "  Project: ${PROJECT}"
  echo "  Experiment mode: ${experiment_mode}"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY RUN] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    return 0
  fi

  "${CMD[@]}"
  sleep 1
}

CONFIGS=()
if [[ "$RUN_ALL" == "1" ]]; then
  while IFS= read -r config_yaml; do
    CONFIGS+=("$config_yaml")
  done < <(all_transformer_configs)
else
  CONFIGS+=("$(resolve_config "$MODEL_OR_CONFIG")")
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No transformer configs found in ultralytics/cfg/models/transformer" >&2
  exit 1
fi

echo "Submitting ${#CONFIGS[@]} transformer segment job(s) for ${MS_DESC} (${MS_TAG}), IMGSZ=${IMGSZ}, OVERLAP=${OVERLAP}, KEEP_FRAC=${KEEP_FRAC}, BATCH=${BATCH}"
for config_yaml in "${CONFIGS[@]}"; do
  submit_config "$config_yaml"
done
