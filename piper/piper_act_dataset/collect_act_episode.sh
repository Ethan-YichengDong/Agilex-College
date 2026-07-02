#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-datasets/piper_act_collection}"
TASK="${TASK:-}"
DURATION="${DURATION:-30}"
EPISODE_LEN="${EPISODE_LEN:-}"
FPS="${FPS:-50}"
PAIR_MODE="${PAIR_MODE:-single}"
ACTION_SOURCE="${ACTION_SOURCE:-slave_next_qpos}"
LEFT_SLAVE_CAN="${LEFT_SLAVE_CAN:-can0}"
RIGHT_SLAVE_CAN="${RIGHT_SLAVE_CAN:-}"
LEFT_MASTER_CAN="${LEFT_MASTER_CAN:-}"
RIGHT_MASTER_CAN="${RIGHT_MASTER_CAN:-}"
CAMERA_NAME="${CAMERA_NAME:-cam_high}"
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
CAMERAS="${CAMERAS:-${CAMERA_NAME}=${CAMERA_DEVICE}}"
NO_CAMERA="${NO_CAMERA:-1}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-240}"
EXPORT_PREVIEW="${EXPORT_PREVIEW:-1}"
EXPORT_VIDEO="${EXPORT_VIDEO:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
NO_ASK_KEEP="${NO_ASK_KEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
  "--dataset-dir" "${DATASET_DIR}"
  "--duration" "${DURATION}"
  "--fps" "${FPS}"
  "--pair-mode" "${PAIR_MODE}"
  "--action-source" "${ACTION_SOURCE}"
  "--left-slave-can" "${LEFT_SLAVE_CAN}"
  "--image-width" "${IMAGE_WIDTH}"
  "--image-height" "${IMAGE_HEIGHT}"
)

if [[ -n "${TASK}" ]]; then
  ARGS+=("--task" "${TASK}")
fi
if [[ -n "${EPISODE_LEN}" ]]; then
  ARGS+=("--episode-len" "${EPISODE_LEN}")
fi
if [[ -n "${RIGHT_SLAVE_CAN}" ]]; then
  ARGS+=("--right-slave-can" "${RIGHT_SLAVE_CAN}")
fi
if [[ -n "${LEFT_MASTER_CAN}" ]]; then
  ARGS+=("--left-master-can" "${LEFT_MASTER_CAN}")
fi
if [[ -n "${RIGHT_MASTER_CAN}" ]]; then
  ARGS+=("--right-master-can" "${RIGHT_MASTER_CAN}")
fi
if [[ "${NO_CAMERA}" == "1" ]]; then
  CAMERAS=""
fi
if [[ -n "${CAMERAS}" ]]; then
  for camera in ${CAMERAS}; do
    ARGS+=("--camera" "${camera}")
  done
fi
if [[ "${EXPORT_PREVIEW}" == "1" ]]; then
  ARGS+=("--export-preview")
fi
if [[ "${EXPORT_VIDEO}" == "1" ]]; then
  ARGS+=("--export-video")
fi
if [[ "${SKIP_PREFLIGHT}" == "1" ]]; then
  ARGS+=("--skip-preflight")
fi
if [[ "${NO_ASK_KEEP}" == "1" ]]; then
  ARGS+=("--no-ask-keep")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

echo "Collect command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/collect_episode.py" "${ARGS[@]}" "$@"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/collect_episode.py "${ARGS[@]}" "$@"
