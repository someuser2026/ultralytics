#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RCNN_TASK="segment"
export DEFAULT_MODEL_ALIAS="cascade-mask-rcnn-smallobj"
export DEFAULT_CONFIG_PATH="ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml"
export DEFAULT_PROJECT="rcnn_segment"
export SCRIPT_PUBLIC_NAME="jobs/train/hpc/bash_scripts_seg/submit_cascade_mask_rcnn.sh"

resolve_config_alias() {
  case "$1" in
    cascade-mask-rcnn-smallobj|cascade-mask-rcnn-smallobj.yaml|cascade_mask_rcnn_r50_fpn_smallobj|cascade_mask_rcnn_r50_fpn_smallobj.yaml)
      printf 'ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml\n'
      ;;
    cascade-mask-rcnn|cascade-mask-rcnn.yaml|cascade_mask_rcnn_r50_fpn|cascade_mask_rcnn_r50_fpn.yaml)
      printf 'ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn.yaml\n'
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_config_help() {
  cat <<'EOF'
  cascade-mask-rcnn-smallobj
  cascade-mask-rcnn
EOF
}

source "${SCRIPT_DIR}/../bash_scripts_rcnn/submit_rcnn_common.sh"
