#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

# Move configured Piper slave arm(s) back to the zero pose.
#
# Examples:
#   bash piper/piper_act_dataset/park_piper_zero.sh
#   CAN=can1 bash piper/piper_act_dataset/park_piper_zero.sh
#   PAIR_MODE=dual LEFT_SLAVE_CAN=can0 RIGHT_SLAVE_CAN=can1 bash piper/piper_act_dataset/park_piper_zero.sh

PAIR_MODE="${PAIR_MODE:-single}"
CAN="${CAN:-can0}"
LEFT_SLAVE_CAN="${LEFT_SLAVE_CAN:-${CAN}}"
RIGHT_SLAVE_CAN="${RIGHT_SLAVE_CAN:-}"
PARK_SECONDS="${PARK_SECONDS:-5}"
PARK_MOVE_SPEED="${PARK_MOVE_SPEED:-20}"
PARK_GRIPPER_EFFORT="${PARK_GRIPPER_EFFORT:-1000}"
PARK_NO_GRIPPER="${PARK_NO_GRIPPER:-0}"
PARK_TRY_CAN_MODE="${PARK_TRY_CAN_MODE:-1}"
PARK_TIMEOUT="${PARK_TIMEOUT:-5}"
DRY_RUN="${DRY_RUN:-0}"

check_can_interface() {
  local can_name="$1"
  local info
  if ! info="$(ip -details link show "${can_name}" 2>&1)"; then
    echo "Cannot inspect ${can_name}: ${info}" >&2
    echo "Bring it up first, for example:" >&2
    echo "  CAN=${can_name} bash piper/piper_act_dataset/can_up.sh" >&2
    exit 1
  fi
  if [[ "${info}" != *"state UP"* && "${info}" != *"<"*UP*">"* ]]; then
    echo "${can_name} is not UP. Bring it up first:" >&2
    echo "  CAN=${can_name} bash piper/piper_act_dataset/can_up.sh" >&2
    exit 1
  fi
  if [[ "${info}" != *"bitrate 1000000"* ]]; then
    echo "${can_name} is not configured at bitrate 1000000." >&2
    echo "${info}" >&2
    exit 1
  fi
}

ARGS=(
  "--seconds" "${PARK_SECONDS}"
  "--move-speed" "${PARK_MOVE_SPEED}"
  "--gripper-effort" "${PARK_GRIPPER_EFFORT}"
  "--timeout" "${PARK_TIMEOUT}"
)

case "${PAIR_MODE}" in
  single)
    check_can_interface "${LEFT_SLAVE_CAN}"
    ARGS+=("--can" "${LEFT_SLAVE_CAN}")
    ;;
  dual)
    if [[ -z "${LEFT_SLAVE_CAN}" || -z "${RIGHT_SLAVE_CAN}" ]]; then
      echo "PAIR_MODE=dual requires LEFT_SLAVE_CAN and RIGHT_SLAVE_CAN." >&2
      exit 1
    fi
    check_can_interface "${LEFT_SLAVE_CAN}"
    check_can_interface "${RIGHT_SLAVE_CAN}"
    ARGS+=("--can" "${LEFT_SLAVE_CAN}" "--can" "${RIGHT_SLAVE_CAN}")
    ;;
  *)
    echo "PAIR_MODE must be single or dual, got: ${PAIR_MODE}" >&2
    exit 1
    ;;
esac

if [[ "${PARK_NO_GRIPPER}" == "1" ]]; then
  ARGS+=("--no-gripper")
fi
if [[ "${PARK_TRY_CAN_MODE}" == "1" ]]; then
  ARGS+=("--try-can-mode")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

echo "Park command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/park_piper_zero.py" "${ARGS[@]}" "$@"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/park_piper_zero.py "${ARGS[@]}" "$@"
