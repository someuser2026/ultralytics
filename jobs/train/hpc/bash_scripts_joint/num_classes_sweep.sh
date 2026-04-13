#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/timm_paths.sh"

MODE="${1:-both}"
IMGSZ="${2:-448}"
MS_OPT="${3:-0}"
BATCH_OBB="${4:-8}"
BATCH_SEG="${5:-8}"
EPOCHS="${6:-100}"
SEED="${7:-0}"

[[ "${IMGSZ}" =~ ^[0-9]+$ && "${IMGSZ}" -ge 32 ]] || { echo "IMGSZ must be integer >= 32"; exit 1; }

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MS_CODE=""
MS_TAG=""
MS_DESC=""
CLS_TAG=""

resolve_ms() {
  local opt="$1"
  case "$opt" in
    0|rgb100|rgb)                   MS_CODE="0";   MS_TAG="RGB100";    MS_DESC="Planet RGB (100 only)";                 CLS_TAG="1cls" ;;
    01|rgbma|rgbmax|rgbmaxarea)     MS_CODE="01";  MS_TAG="RGBMA";     MS_DESC="Planet RGB (maxarea5000)";              CLS_TAG="1cls" ;;
    003|rgb10075s|10075-single)     MS_CODE="003"; MS_TAG="RGB10075S"; MS_DESC="Planet RGB (100+75 collapsed)";         CLS_TAG="1cls" ;;
    1|nir|msnir)                    MS_CODE="1";   MS_TAG="NIR";       MS_DESC="Planet RGB+NIR (analytic_udm2)";        CLS_TAG="1cls" ;;
    11|nirma|nirmax|nirmaxarea)     MS_CODE="11";  MS_TAG="NIRMA";     MS_DESC="Planet RGB+NIR (maxarea5000)";          CLS_TAG="1cls" ;;
    2|pn100|planet_nearmap)         MS_CODE="2";   MS_TAG="PN100";     MS_DESC="Planet+Nearmap (100 only)";             CLS_TAG="1cls" ;;
    21|pn10075s|pn-10075s)          MS_CODE="21";  MS_TAG="PN10075S";  MS_DESC="Planet+Nearmap (100+75 collapsed)";     CLS_TAG="1cls" ;;
    3|nm100|nearmap)                MS_CODE="3";   MS_TAG="NM100";     MS_DESC="Nearmap only (100 only)";               CLS_TAG="1cls" ;;
    001|rgb10075m|10075-multi)      MS_CODE="001"; MS_TAG="RGB10075M"; MS_DESC="Planet RGB (100+75 multi-class)";       CLS_TAG="2cls" ;;
    002|rgball|all)                 MS_CODE="002"; MS_TAG="RGBALL";    MS_DESC="Planet RGB (100+75+50+30 all classes)"; CLS_TAG="4cls" ;;
    *)
      echo "[ERROR] Unknown MS_OPT='${opt}'" >&2
      exit 1
      ;;
  esac
}

submit_job() {
  local task="$1"
  local cfg="$2"
  local freeze="$3"
  local batch="$4"
  local project="$5"
  local extra_params="$6"
  local label
  local ts
  label="$(basename "${cfg}" .yaml)"
  ts="$(date +%m%d-%H%M%S)"
  local job_name="${label}_${CLS_TAG}_${MS_TAG}"
  local varlist="TASK=${task},IMGSZ=${IMGSZ},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WANDB=true,PROJECT=${project},SEED=${SEED},FREEZE=${freeze},EXPERIMENT_MODE=${ts}_${job_name},CONFIG_YAML=${cfg}${extra_params}"
  echo "→ ${job_name}  [FREEZE=${freeze}]  [${CLS_TAG}]  [MS=${MS_TAG}]"
  qsub -V -v "${varlist}" -N "${job_name}" "${PBS_SCRIPT}"
  sleep 1
}

resolve_ms "${MS_OPT}"

declare -a OBB_JOBS=(
  "$(timm_obb_final_augfpn_config yolo11x "${CLS_TAG}")|0"
  "$(timm_obb_final_augfpn_config yolo12x "${CLS_TAG}")|0"
  "$(timm_obb_final_augfpn_config hrnet_w32 "${CLS_TAG}")|0"
  "$(timm_obb_final_augfpn_config dinov3_7_12_17_22 "${CLS_TAG}")|1"
)

declare -a SEG_JOBS=(
  "$(timm_segment_final_yolo_neck_config yolo11x yolo11x "${CLS_TAG}")|0"
  "$(timm_segment_final_yolo_neck_config yolo12x yolo12x "${CLS_TAG}")|0"
  "$(timm_segment_final_yolo_neck_config hrnet_w32 yolo11x "${CLS_TAG}")|0"
  "$(timm_segment_final_yolo_neck_config hrnet_w32 yolo12x_mlp2 "${CLS_TAG}")|0"
  "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo11x "${CLS_TAG}")|1"
  "$(timm_segment_final_yolo_neck_config dinov3_7_12_17_22 yolo12x "${CLS_TAG}")|1"
)

for row in "${OBB_JOBS[@]}"; do
  IFS='|' read -r cfg _freeze <<< "${row}"
  timm_require_file "${cfg}"
done
for row in "${SEG_JOBS[@]}"; do
  IFS='|' read -r cfg _freeze <<< "${row}"
  timm_require_file "${cfg}"
done

TOTAL=0
case "${MODE}" in
  both) TOTAL=$(( ${#OBB_JOBS[@]} + ${#SEG_JOBS[@]} )) ;;
  obb) TOTAL=${#OBB_JOBS[@]} ;;
  segment) TOTAL=${#SEG_JOBS[@]} ;;
  *)
    echo "[ERROR] Unsupported mode: ${MODE}" >&2
    echo "Modes: both, obb, segment"
    exit 1
    ;;
esac

echo "Submitting ${TOTAL} jobs"
echo "MODE=${MODE} | IMGSZ=${IMGSZ} | MS_OPT=${MS_OPT} -> MULTISPECTRAL=${MS_CODE} (${MS_TAG}) | CLS_TAG=${CLS_TAG}"
echo "DATA: ${MS_DESC}"

COUNT=0
if [[ "${MODE}" == "both" || "${MODE}" == "obb" ]]; then
  for row in "${OBB_JOBS[@]}"; do
    IFS='|' read -r cfg freeze <<< "${row}"
    COUNT=$((COUNT + 1))
    printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
    submit_job "obb" "${cfg}" "${freeze}" "${BATCH_OBB}" "backbone_sweep_final" ",CLAHE_P=0.25,MOTION_BLUR_P=0.0,MULTI_SPEC_NOISE_P=0.50,UNSHARP_P=0.0,GAUSSIAN_BLUR_P=0.0"
  done
fi

if [[ "${MODE}" == "both" || "${MODE}" == "segment" ]]; then
  for row in "${SEG_JOBS[@]}"; do
    IFS='|' read -r cfg freeze <<< "${row}"
    COUNT=$((COUNT + 1))
    printf '[%02d/%02d] ' "${COUNT}" "${TOTAL}"
    submit_job "segment" "${cfg}" "${freeze}" "${BATCH_SEG}" "backbone_sweep_segment_final_dice1" ",CLAHE_P=0.0,MOTION_BLUR_P=0.50,MULTI_SPEC_NOISE_P=0.10,UNSHARP_P=0.50,SDICE=1,GAUSSIAN_BLUR_P=0.25"
  done
fi

echo "Done. Monitor with: qstat -u \"$USER\""
