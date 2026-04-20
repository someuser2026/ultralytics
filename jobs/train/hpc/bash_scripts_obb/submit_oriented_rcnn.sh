#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RCNN_TASK="obb"
export DEFAULT_MODEL_ALIAS="oriented-rcnn-smallobj"
export DEFAULT_CONFIG_PATH="ultralytics/cfg/models/rcnn/oriented_rcnn_r50_fpn_le90_smallobj.yaml"
export DEFAULT_PROJECT="rcnn_obb"
export SCRIPT_PUBLIC_NAME="jobs/train/hpc/bash_scripts_obb/submit_oriented_rcnn.sh"
export ANGLE_MODE="${ANGLE_MODE:-le90}"

resolve_config_alias() {
  case "$1" in
    oriented-rcnn-smallobj|oriented-rcnn-smallobj.yaml|oriented_rcnn_r50_fpn_le90_smallobj|oriented_rcnn_r50_fpn_le90_smallobj.yaml)
      printf 'ultralytics/cfg/models/rcnn/oriented_rcnn_r50_fpn_le90_smallobj.yaml\n'
      ;;
    oriented-rcnn|oriented-rcnn.yaml|oriented_rcnn_r50_fpn_le90|oriented_rcnn_r50_fpn_le90.yaml)
      printf 'ultralytics/cfg/models/rcnn/oriented_rcnn_r50_fpn_le90.yaml\n'
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_config_help() {
  cat <<'EOF'
  oriented-rcnn-smallobj
  oriented-rcnn
EOF
}

source "${SCRIPT_DIR}/../bash_scripts_rcnn/submit_rcnn_common.sh"
