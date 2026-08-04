#!/usr/bin/env bash
set -euo pipefail

# DINOv3 + YOLO11x image-size ablation for a manually selected dataset.
# Usage:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_imgsz_ablation.sh <DATASET> [TASK] [BATCH_OBB] [BATCH_SEG] [EPOCHS] [SEED] [DRY_RUN]
#
# DATASET: r_sc, r_sc_mc_1c, r_sc_mc_2c, pn_sc, or pn_sc_mc_1c
# TASK: all (default), obb, or segment
#
# Example:
#   bash jobs/train/hpc/bash_scripts_joint/submit_dinov3_yolo11x_imgsz_ablation.sh pn_sc_mc_1c all 8 8 100 0 1

DATASET="${1:-}"
TASK_SET="${2:-all}"
BATCH_OBB="${3:-8}"
BATCH_SEG="${4:-8}"
EPOCHS="${5:-100}"
SEED="${6:-0}"
DRY_RUN="${7:-0}"

PBS_SCRIPT="jobs/train/hpc/planet_full.pbs"
RUN_TAG="${RUN_TAG:-$(date +%m%d-%H%M%S)}"
WORKERS="${WORKERS:-1}"
IMGSIZES=(224 448 896)

case "$DATASET" in
  r_sc)          MS_CODE="0";   MS_TAG="RSC";     CLASSES="1" ;;
  r_sc_mc_1c)    MS_CODE="003"; MS_TAG="RSCMC1";  CLASSES="1" ;;
  r_sc_mc_2c)    MS_CODE="001"; MS_TAG="RSCMC2";  CLASSES="2" ;;
  pn_sc)         MS_CODE="2";   MS_TAG="PNSC";    CLASSES="1" ;;
  pn_sc_mc_1c)   MS_CODE="21";  MS_TAG="PNSCMC1"; CLASSES="1" ;;
  *)
    echo "DATASET must be one of: r_sc, r_sc_mc_1c, r_sc_mc_2c, pn_sc, pn_sc_mc_1c"
    exit 1
    ;;
esac

case "$TASK_SET" in all|obb|segment) ;; *) echo "TASK must be all, obb, or segment"; exit 1 ;; esac
[[ "$BATCH_OBB" =~ ^[0-9]+$ && "$BATCH_OBB" -ge 1 ]] || { echo "BATCH_OBB must be an integer >= 1"; exit 1; }
[[ "$BATCH_SEG" =~ ^[0-9]+$ && "$BATCH_SEG" -ge 1 ]] || { echo "BATCH_SEG must be an integer >= 1"; exit 1; }
[[ "$EPOCHS" =~ ^[0-9]+$ && "$EPOCHS" -ge 1 ]] || { echo "EPOCHS must be an integer >= 1"; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; exit 1; }
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1"; exit 1; }
[[ -f "$PBS_SCRIPT" ]] || { echo "Missing PBS script: $PBS_SCRIPT"; exit 1; }

OBB_CONFIG="ultralytics/cfg/models/timm/obb/final/augfpn/transformer/dinov3_7_12_17_22/${CLASSES}cls/dinov3_7_12_17_22-augfpn_512c-obb.yaml"
SEG_CONFIG="ultralytics/cfg/models/timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/${CLASSES}cls/dinov3_7_12_17_22-yolo11x-segment.yaml"

NO_AUG=",CLAHE_P=0.0,RAND_GAMMA_P=0.0,UNSHARP_P=0.0,EDGEBOOST_P=0.0,HOMOMORPHIC_P=0.0,GAUSSIAN_BLUR_P=0.0,MOTION_BLUR_P=0.0,ADDITIVE_NOISE_P=0.0,MULTI_SPEC_NOISE_P=0.0,MOSAIC=0.0,MIXUP=0.0,COPY_PASTE=0.0,CLOSE_MOSAIC=0"

submit_job() {
  local task="$1"
  local imgsz="$2"
  local batch project prefix config task_args varlist job_name

  if [[ "$task" == "obb" ]]; then
    batch="$BATCH_OBB"
    project="dino_imgsz_ablation_obb_${MS_TAG}"
    prefix="o"
    config="$OBB_CONFIG"
    task_args=",ANGLE_MODE=le90"
  else
    batch="$BATCH_SEG"
    project="dino_imgsz_ablation_segment_${MS_TAG}"
    prefix="s"
    config="$SEG_CONFIG"
    task_args=",USE_SOFT_IGNORE=false,SDICE=1"
  fi

  [[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }

  job_name="ab_dino_${prefix}_${MS_TAG}_c${imgsz}"
  varlist="TASK=${task},IMGSZ=${imgsz},CHECKPOINT=null,TIME_FLOAT=null,EPOCHS=${EPOCHS},DEVICE=0,EXPERIMENT_MODE=${RUN_TAG}_${job_name},OVERLAP=35,KEEP_FRAC=20,MULTISPECTRAL=${MS_CODE},BATCH=${batch},WORKERS=${WORKERS},CONFIG_YAML=${config},FREEZE=1,SEED=${SEED},WANDB=true,PROJECT=${project}${task_args}${NO_AUG}"

  echo "Submitting ${job_name}: task=${task}, imgsz=${imgsz}, dataset=${DATASET}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN qsub -V -v %q -N %q %q\n' "$varlist" "$job_name" "$PBS_SCRIPT"
  else
    qsub -V -v "$varlist" -N "$job_name" "$PBS_SCRIPT"
    sleep 1
  fi
}

for imgsz in "${IMGSIZES[@]}"; do
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "obb" ]] && submit_job obb "$imgsz"
  [[ "$TASK_SET" == "all" || "$TASK_SET" == "segment" ]] && submit_job segment "$imgsz"
done
