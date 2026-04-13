#!/usr/bin/env bash

TIMM_CFG_ROOT="ultralytics/cfg/models/timm"
YOLO_CFG_ROOT="ultralytics/cfg/models/yolo"

timm_canonical_model() {
  case "$1" in
    seresnet50) echo "seresnetaa50" ;;
    pvt_v2_b2) echo "twins_pcpvt_small" ;;
    maxvit_tiny|maxvit_small) echo "maxvit_small_w7_224" ;;
    dinov3_vitb14) echo "dinov3_7_12_17_22" ;;
    yolov11x) echo "yolo11x" ;;
    yolov11n) echo "yolo11n" ;;
    yolo11x) echo "yolo11x" ;;
    yolo11n) echo "yolo11n" ;;
    yolo12x) echo "yolo12x" ;;
    yolo12n) echo "yolo12n" ;;
    *) echo "$1" ;;
  esac
}

timm_model_family() {
  case "$1" in
    convnext_*|convnextv2_*|convnext_base_dinov3) echo "convnext" ;;
    densenet*) echo "densenet" ;;
    efficientnet*|efficientnetv2_*) echo "efficientnet" ;;
    hrnet_*) echo "hrnet" ;;
    coatnet_*|maxvit_*) echo "hybrid" ;;
    mobilevit_*) echo "mobilevit" ;;
    regnet*) echo "regnet" ;;
    resnet*|resnext*|resnest*|seresnetaa50|skresnext*) echo "resnet" ;;
    deit_*|dinov3_*|pvt_*|swin_*|twins_*) echo "transformer" ;;
    yolo*) echo "yolo" ;;
    *)
      echo "[ERROR] Unknown TIMM model family for: $1" >&2
      return 1
      ;;
  esac
}

timm_require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "[ERROR] Missing config: $path" >&2
    return 1
  }
}

timm_segment_final_panet_config() {
  local model
  local family
  model="$(timm_canonical_model "$1")"
  family="$(timm_model_family "$model")"
  local variant="${2:-flat}"
  local cls_tag="${3:-1cls}"
  echo "${TIMM_CFG_ROOT}/segment/final/panet_adaptive/${family}/${model}/${variant}/${cls_tag}/${model}-panet_adaptive-segment.yaml"
}

timm_segment_final_yolo_neck_config() {
  local model
  local family
  model="$(timm_canonical_model "$1")"
  family="$(timm_model_family "$model")"
  local head="$2"
  local cls_tag="${3:-1cls}"
  local variant="${4:-${cls_tag}}"
  local dir="${variant}"
  case "$variant" in
    1cls|2cls|4cls) ;;
    *)
      dir="${variant}/${cls_tag}"
      ;;
  esac
  echo "${TIMM_CFG_ROOT}/segment/final/yolo_neck/${family}/${model}/${dir}/${model}-${head}-segment.yaml"
}

timm_obb_final_augfpn_config() {
  local model
  local family
  model="$(timm_canonical_model "$1")"
  family="$(timm_model_family "$model")"
  local cls_tag="${2:-1cls}"
  local head_stem="${3:-augfpn_512c}"
  echo "${TIMM_CFG_ROOT}/obb/final/augfpn/${family}/${model}/${cls_tag}/${model}-${head_stem}-obb.yaml"
}

timm_backbone_sweep_config() {
  local task="$1"
  local model
  local family
  model="$(timm_canonical_model "$2")"
  family="$(timm_model_family "$model")"
  local variant="$3"
  local head_stem="${4:-pafpn}"
  local suffix
  case "$task" in
    segment) suffix="segment" ;;
    obb) suffix="obb" ;;
    *)
      echo "[ERROR] Unsupported sweep task: $task" >&2
      return 1
      ;;
  esac
  echo "${TIMM_CFG_ROOT}/${task}/sweeps/backbone/${family}/${model}/${variant}/${model}-${head_stem}-${suffix}.yaml"
}

timm_neck_sweep_config() {
  local task="$1"
  local model
  local family
  model="$(timm_canonical_model "$2")"
  family="$(timm_model_family "$model")"
  local head_stem="$3"
  local variant="${4:-base}"
  local suffix
  case "$task" in
    segment) suffix="segment" ;;
    obb) suffix="obb" ;;
    *)
      echo "[ERROR] Unsupported sweep task: $task" >&2
      return 1
      ;;
  esac
  echo "${TIMM_CFG_ROOT}/${task}/sweeps/neck/${family}/${model}/${variant}/${model}-${head_stem}-${suffix}.yaml"
}

yolo_segment_no_p2_pafpn_config() {
  local model
  model="$(timm_canonical_model "$1")"
  echo "${YOLO_CFG_ROOT}/segment_no_p2/${model}-pafpn-segment.yaml"
}

yolo_obb_pafpn_config() {
  local model
  model="$(timm_canonical_model "$1")"
  echo "${YOLO_CFG_ROOT}/obb/${model}-pafpn-obb.yaml"
}
