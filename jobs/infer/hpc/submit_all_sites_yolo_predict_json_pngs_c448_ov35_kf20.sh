#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SUBMIT_SCRIPT="${REPO_ROOT}/jobs/infer/hpc/submit_yolo_site_predict_json.sh"

CHECKPOINT="/srv/scratch/z5428587/runs/cuda/segment/imgsz_448/yolo/mamba_joint_seg_PN10075S_c448/0507-154304_s_mamba_hr_PN10075S_c448/weights/best.pt"
IMG_DIR="visual/pngs_c448_ov35_kf20"
IMGSZ="${IMGSZ:-448}"
CONF="${CONF:-0.01}"
DEVICE="${DEVICE:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOB_BATCHING="1"
export JOB_BATCH_SIZE="${JOB_BATCH_SIZE:-5000}"

cd "${REPO_ROOT}"

submit_site() {
  local site_name="$1"
  bash "${SUBMIT_SCRIPT}" \
    "${CHECKPOINT}" \
    "${site_name}" \
    "${IMG_DIR}" \
    "${IMGSZ}" \
    "${CONF}" \
    "${DEVICE}" \
    "${DRY_RUN}" \
    "${JOB_BATCHING}"
}

# submit_site "Tairua"
# submit_site "Bondi"
# submit_site "Treachery"
# submit_site "TrucVert"
# submit_site "Arrifana"
# submit_site "Manly"
# submit_site "Narra"
# submit_site "NorthTorre"
# submit_site "DiscoveryBay"
# submit_site "GunyahLeft"
# submit_site "GunyahLeftmost"
# submit_site "GunyahRight"
# submit_site "IndianaDunes"
# submit_site "NetherlandCentralCoastMiddle"
# submit_site "PalmCoveNorth"
# submit_site "Runkerry"
# submit_site "ScrippsSouth"
# submit_site "Turimetta"
# submit_site "Virginia"
# submit_site "WasagaNorth"
# submit_site "BravaNorth"
# submit_site "BravaSouth"
# submit_site "BroadwaterMiddle"
