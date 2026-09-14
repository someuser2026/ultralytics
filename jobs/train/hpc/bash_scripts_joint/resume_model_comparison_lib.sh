#!/usr/bin/env bash

# Shared selector for the RSCMC1 and PNSCMC1 model-comparison resume wrappers.

resume_model_comparison() {
  local entry alias task imgsz project
  local selected=0
  local jobs=(
    "yolo11_obb|obb"
    "yolo12_obb|obb"
    "yolo26_obb|obb"
    "dino_obb|obb"
    "mamba_yolo_obb|obb"
    "mamba_hr_obb|obb"
    "hr32_obb|obb"
    "rfrcnn|obb"
    "orcnn_leg|obb"
    "rfcos_r50|obb"
    "rfcos_leg|obb"
    "rhino|obb"
    "yolo11_seg|segment"
    "yolo12_seg|segment"
    "yolo26_seg|segment"
    "dino_seg|segment"
    "mamba_yolo_seg|segment"
    "mamba_hr_seg|segment"
    "cascade|segment"
    "mask_rcnn|segment"
    "pointrend|segment"
    "mask2former|segment"
    "mask2former_hrnet|segment"
  )

  if [[ "$MODELS" == "--list" ]]; then
    echo "OBB: yolo11_obb yolo12_obb yolo26_obb dino_obb mamba_yolo_obb mamba_hr_obb hr32_obb rfrcnn orcnn_leg rfcos_r50 rfcos_leg rhino"
    echo "SEG: yolo11_seg yolo12_seg yolo26_seg dino_seg mamba_yolo_seg mamba_hr_seg cascade mask_rcnn pointrend mask2former mask2former_hrnet"
    return
  fi

  [[ "$IMGSZ_OBB" =~ ^[0-9]+$ && "$IMGSZ_OBB" -ge 32 ]] || { echo "IMGSZ_OBB must be an integer >= 32"; return 1; }
  [[ "$IMGSZ_SEG" =~ ^[0-9]+$ && "$IMGSZ_SEG" -ge 32 ]] || { echo "IMGSZ_SEG must be an integer >= 32"; return 1; }
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer"; return 1; }

  source jobs/train/hpc/bash_scripts_joint/resume_incomplete_runs_lib.sh
  resume_init

  for entry in "${jobs[@]}"; do
    IFS='|' read -r alias task <<< "$entry"
    if [[ "$MODELS" == "all" || "$MODELS" == "$task" || ",${MODELS}," == *",${alias},"* ]]; then
      if [[ "$task" == "obb" ]]; then
        imgsz="$IMGSZ_OBB"
        project="$PROJECT_OBB"
      else
        imgsz="$IMGSZ_SEG"
        project="$PROJECT_SEG"
      fi
      resume_consider_run "$task" "$imgsz" "$project" "${alias}_seed${SEED}"
      selected=$((selected + 1))
    fi
  done

  [[ "$selected" -gt 0 ]] || { echo "No runnable models selected. Use --list for valid aliases."; return 1; }
  resume_submit_selected
}
