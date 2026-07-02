#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/data/raw}"
EPISODE="${1:-}"
CAN="${CAN:-can0}"
SOURCE="${SOURCE:-qpos}"
SIDE="${SIDE:-left}"
MAX_FRAMES="${MAX_FRAMES:-100}"
START_FRAME="${START_FRAME:-0}"
SPEED_SCALE="${SPEED_SCALE:-0.5}"
MOVE_SPEED="${MOVE_SPEED:-20}"
NO_GRIPPER="${NO_GRIPPER:-1}"
MOVE_TO_START="${MOVE_TO_START:-0}"
TRY_CAN_MODE="${TRY_CAN_MODE:-0}"
YES="${YES:-0}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${EPISODE}" ]]; then
  EPISODE="$(latest_episode "${DATASET_DIR}")"
fi
if [[ -z "${EPISODE}" ]]; then
  echo "No episode provided and no episode_*.hdf5 found under ${DATASET_DIR}." >&2
  echo "Usage: $0 path/to/episode_N.hdf5" >&2
  exit 1
fi

ARGS=(
  "${EPISODE}"
  "--can" "${CAN}"
  "--source" "${SOURCE}"
  "--side" "${SIDE}"
  "--start-frame" "${START_FRAME}"
  "--speed-scale" "${SPEED_SCALE}"
  "--move-speed" "${MOVE_SPEED}"
)

if [[ -n "${MAX_FRAMES}" ]]; then
  ARGS+=("--max-frames" "${MAX_FRAMES}")
fi
if [[ "${NO_GRIPPER}" == "1" ]]; then
  ARGS+=("--no-gripper")
fi
if [[ "${MOVE_TO_START}" == "1" ]]; then
  ARGS+=("--move-to-start")
fi
if [[ "${TRY_CAN_MODE}" == "1" ]]; then
  ARGS+=("--try-can-mode")
fi
if [[ "${YES}" == "1" ]]; then
  ARGS+=("--yes")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

echo "Replay command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/play_episode_piper.py" "${ARGS[@]}" "${@:2}"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/play_episode_piper.py "${ARGS[@]}" "${@:2}"
