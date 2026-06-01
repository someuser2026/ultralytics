#!/usr/bin/env bash
set -euo pipefail

# Submit the selected Mamba-HRNet segment/cascade jobs on the sh-lw-only dataset variant.
#
# Usage:
#   bash jobs/train/hpc/bash_scripts_seg/submit_mamba_hrnet_shore_lw_only_segment_models.sh [BATCH] [EPOCHS] [SEED] [DRY_RUN] [WORKERS]
#
# Defaults are inherited from submit_mamba_hrnet_shore_lw_input_segment_models.sh:
#   BATCH=4, EPOCHS=100, SEED=0, DRY_RUN=0, WORKERS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DATA_YAML:-}" ]]; then
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH must be set unless DATA_YAML is provided" >&2
    exit 1
  fi
  DATA_YAML="${SCRATCH}/data_processed/Global/Annotated/variants/segment/planet_full_c448_ov35_kf20_10075-single_sh-lw_seed0/data.yaml"
  export DATA_YAML
fi

exec "${SCRIPT_DIR}/submit_mamba_hrnet_shore_lw_input_segment_models.sh" "$@"
