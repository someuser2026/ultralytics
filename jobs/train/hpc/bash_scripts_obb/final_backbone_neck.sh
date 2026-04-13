#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../bash_scripts_joint/timm_paths.sh"
# Usage: bash backbone_sweep_loop.sh [IMGSZ] [BATCH] [MULTISPECTRAL_OPT]
IMGSZ="${1:-224}"
BATCH="${2:-16}"
MULTISPECTRAL_OPT="${3:-0}"

# Basic validation
[[ "$IMGSZ" =~ ^[0-9]+$ && "$IMGSZ" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"

# ------------------------------------------------------------------------------
# MULTISPECTRAL option resolver
#
# You can pass either the legacy numeric codes OR readable aliases.
#
# Planet RGB (planet_full)
#   0    | rgb100      -> .../planet_full_..._seed{seed}                (100 only)
#   01   | rgbma       -> .../planet_full_..._maxarea5000_seed{seed}
#   001  | rgb10075m   -> .../planet_full_..._10075-multi_seed{seed}
#   002  | rgball      -> .../planet_full_..._all_seed{seed}
#   003  | rgb10075s   -> .../planet_full_..._10075-single_seed{seed}
#
# Planet analytic (planet_full_analytic_udm2)
#   1    | nir         -> .../planet_full_analytic_udm2_..._seed{seed}
#   11   | nirma       -> .../planet_full_analytic_udm2_..._maxarea5000_seed{seed}
#
# Planet + Nearmap (planet_nearmap)
#   2    | pn100       -> .../planet_nearmap_..._seed{seed}             (100 only)
#   21   | pn10075s    -> .../planet_nearmap_..._10075-single_seed{seed} (100+75 collapsed)
#
# Nearmap only (nearmap)
#   3    | nm100       -> .../nearmap_..._seed{seed}                    (100 only)
#
# This script:
#   - passes MULTISPECTRAL=<numeric code> to PBS (backward-compatible)
#   - appends a short modifier tag to job name + experiment mode (readable at glance)
# ------------------------------------------------------------------------------
MS_CODE=""
MS_TAG=""
MS_DESC=""
CLS_TAG=""

resolve_ms() {
  local opt="$1"
  case "$opt" in
    # Planet RGB
    0|rgb|rgb100)
      MS_CODE="0";   MS_TAG="RGB100";     MS_DESC="Planet RGB (100 only)"; CLS_TAG="1cls" ;;
    01|rgbma|rgb-max|rgbmax|rgbmaxarea)
      MS_CODE="01";  MS_TAG="RGBMA";      MS_DESC="Planet RGB (maxarea5000)"; CLS_TAG="1cls" ;;
    001|rgb10075m|rgb-10075m|10075-multi)
      MS_CODE="001"; MS_TAG="RGB10075M";  MS_DESC="Planet RGB (100+75 multi-class)"; CLS_TAG="2cls" ;;
    002|rgball|rgb-all|all)
      MS_CODE="002"; MS_TAG="RGBALL";     MS_DESC="Planet RGB (100+75+50+30 all classes)"; CLS_TAG="4cls" ;;
    003|rgb10075s|rgb-10075s|10075-single)
      MS_CODE="003"; MS_TAG="RGB10075S";  MS_DESC="Planet RGB (100+75 collapsed)"; CLS_TAG="1cls" ;;

    # Planet analytic (RGB+NIR)
    1|nir|msnir)
      MS_CODE="1";   MS_TAG="NIR";        MS_DESC="Planet RGB+NIR (analytic_udm2)"; CLS_TAG="1cls" ;;
    11|nirma|nirmax|nir-max|nirmaxarea)
      MS_CODE="11";  MS_TAG="NIRMA";      MS_DESC="Planet RGB+NIR (maxarea5000)"; CLS_TAG="1cls" ;;

    # Planet + Nearmap
    2|pn|pn100|planet_nearmap)
      MS_CODE="2";   MS_TAG="PN100";      MS_DESC="Planet+Nearmap (100 only)"; CLS_TAG="1cls" ;;
    21|pn10075s|pn-10075s)
      MS_CODE="21";  MS_TAG="PN10075S";   MS_DESC="Planet+Nearmap (100+75 collapsed)"; CLS_TAG="1cls" ;;

    # Nearmap only
    3|nm|nm100|nearmap)
      MS_CODE="3";   MS_TAG="NM100";      MS_DESC="Nearmap only (100 only)"; CLS_TAG="1cls" ;;

    *)
      echo "[ERROR] Unknown MULTISPECTRAL_OPT='$opt'"
      echo "Valid options (code or alias):"
      echo "  0|rgb100, 01|rgbma, 001|rgb10075m, 002|rgball, 003|rgb10075s,"
      echo "  1|nir, 11|nirma, 2|pn100, 21|pn10075s, 3|nm100"
      exit 1 ;;
  esac
}
resolve_ms "$MULTISPECTRAL_OPT"

# Common env (task/config family stays OBB; tweak if you switch task)
COMMON_PARAMS="TASK=obb,IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=100,DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${BATCH},CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0,WANDB=true,PROJECT=backbone_sweep_final"

submit_job() {
  local label="$1" config_yaml="$2" freeze="$3"
  local ts; ts="$(date +%m%d-%H%M%S)"

  # Add MS_TAG into job name and experiment mode for quick identification
  local job_name="${label}_${MS_TAG}"
  local exp_mode="${ts}_${label}_${MS_TAG}"

  local VARLIST="${COMMON_PARAMS},FREEZE=${freeze},EXPERIMENT_MODE=${exp_mode},CONFIG_YAML=${config_yaml}"
  echo "→ ${label}  [FREEZE=${freeze}]  [MS=${MS_TAG}]  (${MS_DESC})"
  qsub -V -v "${VARLIST}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

# Format: "label|model|freeze"
JOBS=(
#   # CNN BACKBONES — ResNet family

#   # CNN BACKBONES — EfficientNet family

#   # CNN BACKBONES — YOLO family
  "yolo12x|yolo12x|0"

#   # CNN BACKBONES — ConvNeXt family

#   # CNN BACKBONES — RegNet family

#   # CNN BACKBONES — MobileNet family

#   # CNN BACKBONES — DenseNet family

#   # CNN BACKBONES — HRNet family
  "hrnet_w32_rgb|hrnet_w32|0"

#   # VISION TRANSFORMERS — Swin family

#   # VISION TRANSFORMERS — PVT family

#   # VISION TRANSFORMERS — DeiT family

#   # VISION TRANSFORMERS — BEiT family

#   # VISION TRANSFORMERS — MaxViT family

#   # VISION TRANSFORMERS — EfficientViT family

#   # VISION TRANSFORMERS — MobileViT family

#   # HYBRID — CoAtNet family

#   # FOUNDATION — DINOv3 (frozen backbone)
  "dinov3_vitb14_rgb|dinov3_vitb14|1"
)

# Summary
ACTIVE_COUNT=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  ACTIVE_COUNT=$((ACTIVE_COUNT+1))
done

echo "============================================================"
echo "Submitting ${ACTIVE_COUNT} jobs"
echo "IMGSZ=${IMGSZ}  BATCH=${BATCH}"
echo "MULTISPECTRAL_OPT=${MULTISPECTRAL_OPT} -> MULTISPECTRAL=${MS_CODE}  TAG=${MS_TAG}"
echo "DATA: ${MS_DESC}"
echo "PBS:  ${PBS_SCRIPT}"
echo "============================================================"

i=0
for job in "${JOBS[@]}"; do
  [[ -z "$job" || "$job" =~ ^[[:space:]]*# ]] && continue
  IFS='|' read -r label model freeze <<< "${job}"
  config_yaml="$(timm_obb_final_augfpn_config "${model}" "${CLS_TAG}")"
  timm_require_file "${config_yaml}"
  i=$((i+1))
  printf '[%02d/%02d] ' "${i}" "${ACTIVE_COUNT}"
  submit_job "${label}" "${config_yaml}" "${freeze}"
done

echo "Done. Monitor with: qstat -u \"$USER\""
