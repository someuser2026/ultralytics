#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_benchmark_models.sh [MODELS] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [DRY_RUN]
#
# MODELS may be:
#   all                         submit all runnable models (default)
#   obb                         submit all OBB models
#   segment                     submit all segmentation models
#   model1,model2               submit selected model aliases from --list
#
# Examples:
#   bash jobs/train/hpc/bash_scripts_joint/submit_benchmark_models.sh all 8 8 100 0 1
#   bash jobs/train/hpc/bash_scripts_joint/submit_benchmark_models.sh pointrend,mask2former,rhino 8 8 100 0 0
#   bash jobs/train/hpc/bash_scripts_joint/submit_benchmark_models.sh --list
#
# Optional environment overrides: IMGSZ_OBB, IMGSZ_SEG, MULTISPECTRAL, DATASET_NAME, PROJECT_OBB, PROJECT_SEG,
# OPTIMIZER, LR0, LRF, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP_NORM

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
MODELS="${1:-all}"
BATCH_OBB="${2:-8}"
BATCH_SEG="${3:-8}"
EPOCHS="${4:-100}"
SEED="${5:-0}"
DRY_RUN="${6:-0}"

IMGSZ_OBB="${IMGSZ_OBB:-448}"
IMGSZ_SEG="${IMGSZ_SEG:-448}"
MULTISPECTRAL="${MULTISPECTRAL:-21}"
DATASET_NAME="${DATASET_NAME:-}"
WORKERS="${WORKERS:-1}"
BATCH_RHINO="${BATCH_RHINO:-4}"
BATCH_MASK2FORMER="${BATCH_MASK2FORMER:-4}"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"

# Stable transformer-style optimization for the RHINO and standard Mamba-YOLO OBB models.
# Mamba-HR OBB keeps the generic Ultralytics defaults because its existing run is stable.
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.0001}"
LRF="${LRF:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.01}"

if [[ -z "$DATASET_NAME" ]]; then
  case "$MULTISPECTRAL" in
    0) DATASET_NAME="RSC" ;;
    003) DATASET_NAME="RSCMC1" ;;
    001) DATASET_NAME="RSCMC2" ;;
    2) DATASET_NAME="PNSC" ;;
    21) DATASET_NAME="PNSCMC1" ;;
    *) DATASET_NAME="dataset_${MULTISPECTRAL}" ;;
  esac
fi
DATASET_SLUG="$(printf '%s' "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')"
PROJECT_OBB="${PROJECT_OBB:-benchmark_${DATASET_SLUG}_imgsz${IMGSZ_OBB}_obb}"
PROJECT_SEG="${PROJECT_SEG:-benchmark_${DATASET_SLUG}_imgsz${IMGSZ_SEG}_segment}"

# alias|task|job_name|config|freeze
JOBS=(
  "yolo11_obb|obb|b_y11_obb|ultralytics/cfg/models/11/yolo11x-obb-1cls.yaml|0"
  "yolo12_obb|obb|b_y12_obb|ultralytics/cfg/models/12/yolo12x-obb-1cls.yaml|0"
  "dino_obb|obb|b_dino_obb|ultralytics/cfg/models/timm/obb/final/augfpn/transformer/dinov3_7_12_17_22/1cls/dinov3_7_12_17_22-augfpn_512c-obb.yaml|1"
  "mamba_yolo_obb|obb|b_mamba_yolo_o|ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml|0"
  "mamba_hr_obb|obb|b_mamba_hr_o|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb.yaml|0"
  "hr32_obb|obb|b_hr32_obb|ultralytics/cfg/models/timm/obb/final/augfpn/hrnet/hrnet_w32/1cls/hrnet_w32-augfpn_512c-obb.yaml|0"
  "rfrcnn|obb|b_rfrcnn|ultralytics/cfg/models/rcnn/rotated_faster_rcnn_r50_fpn_le90_smallobj.yaml|0"
  "orcnn_leg|obb|b_orcnn_leg|ultralytics/cfg/models/rcnn/oriented_rcnn_legnet_small_fpn_le90_smallobj.yaml|0"
  "rfcos_r50|obb|b_rfcos_r50|ultralytics/cfg/models/fcos/rotated_fcos_r50_fpn_le90.yaml|0"
  "rfcos_leg|obb|b_rfcos_leg|ultralytics/cfg/models/legnet/legnet-small-fcos-smallobj.yaml|0"
  "rhino|obb|b_rhino_r50|ultralytics/cfg/models/rhino/rhino-r50-obb.yaml|0"
  "yolo11_seg|segment|b_y11_seg|ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/yolo11x-yolo11x-segment.yaml|0"
  "yolo12_seg|segment|b_y12_seg|ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml|0"
  "dino_seg|segment|b_dino_seg|ultralytics/cfg/models/timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/dinov3_7_12_17_22-yolo11x-segment.yaml|1"
  "mamba_yolo_seg|segment|b_mamba_yolo_s|ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-seg.yaml|0"
  "mamba_hr_seg|segment|b_mamba_hr_s|ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml|0"
  "cascade|segment|b_cascade|ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml|0"
  "mask_rcnn|segment|b_mask_rcnn|ultralytics/cfg/models/rcnn/mask_rcnn_r50_fpn_smallobj.yaml|0"
  "pointrend|segment|b_pointrend|ultralytics/cfg/models/rcnn/pointrend_rcnn_r50_fpn_smallobj.yaml|0"
  "mask2former|segment|b_mask2former|ultralytics/cfg/models/transformer/mask2former-swin-timm-seg.yaml|0"
  "mask2former_hrnet|segment|b_m2f_hr32|ultralytics/cfg/models/transformer/mask2former-hrnet-w32-timm-seg.yaml|0"
)

if [[ "$MODELS" == "--list" ]]; then
  echo "OBB: yolo11_obb yolo12_obb dino_obb mamba_yolo_obb mamba_hr_obb hr32_obb rfrcnn orcnn_leg rfcos_r50 rfcos_leg rhino"
  echo "SEG: yolo11_seg yolo12_seg dino_seg mamba_yolo_seg mamba_hr_seg cascade mask_rcnn pointrend mask2former mask2former_hrnet"
  echo "UNAVAILABLE: roit_leg (listed in the manuscript but no native Ultralytics config exists)"
  exit 0
fi

[[ "$BATCH_OBB" =~ ^[0-9]+$ && "$BATCH_OBB" -ge 1 ]] || { echo "BATCH_OBB must be an integer >= 1"; exit 1; }
[[ "$BATCH_SEG" =~ ^[0-9]+$ && "$BATCH_SEG" -ge 1 ]] || { echo "BATCH_SEG must be an integer >= 1"; exit 1; }
[[ "$IMGSZ_OBB" =~ ^[0-9]+$ && "$IMGSZ_OBB" -ge 32 ]] || { echo "IMGSZ_OBB must be an integer >= 32"; exit 1; }
[[ "$IMGSZ_SEG" =~ ^[0-9]+$ && "$IMGSZ_SEG" -ge 32 ]] || { echo "IMGSZ_SEG must be an integer >= 32"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

selected() {
  local alias="$1"
  local task="$2"
  [[ "$MODELS" == "all" || "$MODELS" == "$task" || ",${MODELS}," == *",${alias},"* ]]
}

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submit_job() {
  local alias="$1"
  local task="$2"
  local job_name="$3"
  local config="$4"
  local freeze="$5"
  local batch imgsz project task_args optimizer_args varlist

  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  if [[ "$task" == "obb" ]]; then
    batch="$BATCH_OBB"
    imgsz="$IMGSZ_OBB"
    project="$PROJECT_OBB"
    task_args=",ANGLE_MODE=le90"
  else
    batch="$BATCH_SEG"
    imgsz="$IMGSZ_SEG"
    project="$PROJECT_SEG"
    task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  fi
  [[ "$alias" == "rhino" ]] && batch="$BATCH_RHINO"
  [[ "$alias" == "mask2former" || "$alias" == "mask2former_hrnet" ]] && batch="$BATCH_MASK2FORMER"

  optimizer_args=""
  if [[ "$alias" == "mamba_yolo_obb" || "$alias" == "rhino" ]]; then
    optimizer_args=",OPTIMIZER=${OPTIMIZER},LR0=${LR0},LRF=${LRF},WEIGHT_DECAY=${WEIGHT_DECAY},WARMUP_EPOCHS=${WARMUP_EPOCHS},GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
  fi

  varlist="TASK=${task},IMGSZ=${imgsz},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${alias},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MULTISPECTRAL},BATCH=${batch},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=${freeze},SEED=${SEED},WANDB=true,PLOTS=false,PROJECT=${project}${task_args}${optimizer_args}${NO_AUG}"

  if [[ -n "$optimizer_args" ]]; then
    echo "Submitting ${alias}: task=${task}, imgsz=${imgsz}, batch=${batch}, config=${config}, optimizer=${OPTIMIZER}, lr0=${LR0}"
  else
    echo "Submitting ${alias}: task=${task}, imgsz=${imgsz}, batch=${batch}, config=${config}"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

submitted=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r alias task job_name config freeze <<< "$entry"
  if selected "$alias" "$task"; then
    submit_job "$alias" "$task" "$job_name" "$config" "$freeze"
    submitted=$((submitted + 1))
  fi
done

if [[ "$MODELS" == "roit_leg" || ",${MODELS}," == *",roit_leg,"* ]]; then
  echo "Skipping roit_leg: no native Ultralytics configuration exists."
fi

[[ "$submitted" -gt 0 ]] || { echo "No runnable models selected. Use --list for valid aliases."; exit 1; }
echo "Processed ${submitted} benchmark job(s)."
