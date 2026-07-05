#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

# Request the Piper leader/follower pair to return to zero using SDK
# ReqMasterArmMoveToHome. This is intended for the shared-can0 master-slave
# setup and does not send MasterSlaveConfig role commands.

CAN="${CAN:-can0}"
MASTER_HOME_MODE="${MASTER_HOME_MODE:-both_zero}"
MASTER_HOME_CYCLE="${MASTER_HOME_CYCLE:-1}"
MASTER_HOME_WAIT="${MASTER_HOME_WAIT:-6}"
MASTER_HOME_RESTORE="${MASTER_HOME_RESTORE:-1}"
MASTER_HOME_CHECK="${MASTER_HOME_CHECK:-0}"
MASTER_HOME_PREFLIGHT="${MASTER_HOME_PREFLIGHT:-1}"
MASTER_HOME_FIRMWARE_TIMEOUT="${MASTER_HOME_FIRMWARE_TIMEOUT:-3}"
MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE="${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE:-1}"
MASTER_HOME_VERIFY_ZERO="${MASTER_HOME_VERIFY_ZERO:-1}"
MASTER_HOME_ZERO_TIMEOUT="${MASTER_HOME_ZERO_TIMEOUT:-8}"
MASTER_HOME_JOINT_TOLERANCE="${MASTER_HOME_JOINT_TOLERANCE:-0.08}"
MASTER_HOME_VERIFY_GRIPPER="${MASTER_HOME_VERIFY_GRIPPER:-0}"
MASTER_HOME_GRIPPER_TOLERANCE="${MASTER_HOME_GRIPPER_TOLERANCE:-0.01}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=("--can" "${CAN}")

if [[ "${MASTER_HOME_CHECK}" == "1" ]]; then
  ARGS+=("--check-support" "--firmware-timeout" "${MASTER_HOME_FIRMWARE_TIMEOUT}")
  if [[ "${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE}" == "1" ]]; then
    ARGS+=("--allow-unknown-firmware")
  fi
else
  if [[ "${MASTER_HOME_PREFLIGHT}" == "1" ]]; then
    ARGS+=("--preflight" "--firmware-timeout" "${MASTER_HOME_FIRMWARE_TIMEOUT}")
    if [[ "${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE}" == "1" ]]; then
      ARGS+=("--allow-unknown-firmware")
    fi
  fi
fi

if [[ "${MASTER_HOME_CHECK}" != "1" && "${MASTER_HOME_CYCLE}" == "1" ]]; then
  ARGS+=("--cycle" "--wait" "${MASTER_HOME_WAIT}")
  if [[ "${MASTER_HOME_RESTORE}" != "1" ]]; then
    ARGS+=("--no-restore")
  fi
  if [[ "${MASTER_HOME_VERIFY_ZERO}" == "1" ]]; then
    ARGS+=("--verify-zero")
    ARGS+=("--zero-timeout" "${MASTER_HOME_ZERO_TIMEOUT}")
    ARGS+=("--joint-tolerance" "${MASTER_HOME_JOINT_TOLERANCE}")
    if [[ "${MASTER_HOME_VERIFY_GRIPPER}" == "1" ]]; then
      ARGS+=("--gripper-tolerance" "${MASTER_HOME_GRIPPER_TOLERANCE}")
    fi
  fi
elif [[ "${MASTER_HOME_CHECK}" != "1" ]]; then
  ARGS+=("--mode" "${MASTER_HOME_MODE}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

echo "Master-home command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/request_master_home.py" "${ARGS[@]}" "$@"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/request_master_home.py "${ARGS[@]}" "$@"
