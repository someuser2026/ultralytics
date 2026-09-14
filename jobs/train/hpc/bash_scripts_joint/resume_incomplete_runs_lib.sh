#!/usr/bin/env bash

# Shared helpers for experiment-specific resume launchers. This file is intended to be sourced.

resume_init() {
  RESUME_PROFILE="${PROFILE:-full}"
  RESUME_TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
  RESUME_DRY_RUN="${DRY_RUN:-0}"
  RESUME_WANDB="${WANDB:-true}"
  RESUME_SUBMITTER="jobs/train/hpc/submit_resume_training.sh"

  case "$RESUME_PROFILE" in full|2hr) ;; *) echo "PROFILE must be full or 2hr" >&2; return 2 ;; esac
  [[ "$RESUME_TARGET_EPOCHS" =~ ^[0-9]+$ && "$RESUME_TARGET_EPOCHS" -ge 1 ]] || {
    echo "TARGET_EPOCHS must be an integer >= 1" >&2
    return 2
  }
  [[ "$RESUME_DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; return 2; }
  [[ -f "$RESUME_SUBMITTER" ]] || { echo "Missing resume submitter: $RESUME_SUBMITTER" >&2; return 1; }

  if [[ -n "${RUNS_ROOT:-}" ]]; then
    RESUME_RUNS_ROOT="$RUNS_ROOT"
  elif [[ -n "${SCRATCH:-}" ]]; then
    RESUME_RUNS_ROOT="${SCRATCH}/runs/cuda"
  else
    echo "Set SCRATCH, or set RUNS_ROOT directly, so completed runs can be located." >&2
    return 2
  fi
  [[ -d "$RESUME_RUNS_ROOT" ]] || { echo "Runs root not found: $RESUME_RUNS_ROOT" >&2; return 1; }

  RESUME_CHECKPOINTS=()
  RESUME_SELECTED=0
  RESUME_COMPLETE=0
  RESUME_MISSING=0
  shopt -s nullglob
}

resume_latest_matching_run() {
  local project_dir="$1"
  local suffix="$2"
  local candidates=("${project_dir}"/*_"${suffix}")
  local candidate marker latest="" latest_marker=""
  local count=0

  for candidate in "${candidates[@]}"; do
    [[ -d "$candidate" ]] || continue
    marker="$candidate"
    [[ -f "${candidate}/results.csv" ]] && marker="${candidate}/results.csv"
    [[ -f "${candidate}/weights/last.pt" ]] && marker="${candidate}/weights/last.pt"
    if [[ -z "$latest" || "$marker" -nt "$latest_marker" ]]; then
      latest="$candidate"
      latest_marker="$marker"
    fi
    count=$((count + 1))
  done

  [[ -n "$latest" ]] || return 1
  if [[ "$count" -gt 1 ]]; then
    echo "Found ${count} runs ending in ${suffix}; using latest: ${latest}" >&2
  fi
  printf '%s\n' "$latest"
}

resume_completed_epochs() {
  awk 'NR > 1 && $0 !~ /^[[:space:]]*$/ { count += 1 } END { print count + 0 }' "$1"
}

resume_consider_run() {
  local task="$1"
  local imgsz="$2"
  local project="$3"
  local suffix="$4"
  local label="${task}/imgsz_${imgsz}/${project}/*_${suffix}"
  local project_dir="${RESUME_RUNS_ROOT}/${task}/imgsz_${imgsz}/yolo/${project}"
  local run_dir results checkpoint completed

  RESUME_SELECTED=$((RESUME_SELECTED + 1))
  if ! run_dir="$(resume_latest_matching_run "$project_dir" "$suffix")"; then
    echo "MISSING ${label}"
    RESUME_MISSING=$((RESUME_MISSING + 1))
    return
  fi

  results="${run_dir}/results.csv"
  checkpoint="${run_dir}/weights/last.pt"
  if [[ ! -f "$results" ]]; then
    echo "MISSING results.csv; cannot determine completion: ${run_dir}"
    RESUME_MISSING=$((RESUME_MISSING + 1))
    return
  fi

  completed="$(resume_completed_epochs "$results")"
  if [[ "$completed" -ge "$RESUME_TARGET_EPOCHS" ]]; then
    echo "COMPLETE ${completed}/${RESUME_TARGET_EPOCHS}: ${run_dir}"
    RESUME_COMPLETE=$((RESUME_COMPLETE + 1))
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "MISSING resumable weights/last.pt (${completed}/${RESUME_TARGET_EPOCHS} epochs): ${run_dir}"
    RESUME_MISSING=$((RESUME_MISSING + 1))
    return
  fi

  echo "RESUME ${completed}/${RESUME_TARGET_EPOCHS}: ${checkpoint}"
  RESUME_CHECKPOINTS+=("$checkpoint")
}

resume_submit_selected() {
  local queued="${#RESUME_CHECKPOINTS[@]}"
  echo "Selected ${RESUME_SELECTED} run(s): ${queued} incomplete, ${RESUME_COMPLETE} complete, ${RESUME_MISSING} missing."
  if [[ "$queued" -eq 0 ]]; then
    echo "No incomplete runs to submit."
    return
  fi

  DRY_RUN="$RESUME_DRY_RUN" WANDB="$RESUME_WANDB" \
    bash "$RESUME_SUBMITTER" "$RESUME_PROFILE" "${RESUME_CHECKPOINTS[@]}"
}
