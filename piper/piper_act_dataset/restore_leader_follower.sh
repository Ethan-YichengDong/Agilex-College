#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

# Restore Piper leader-follower mode after CAN parking, or any time the arms
# need to be put back into teaching mode.

PAIR_MODE="${PAIR_MODE:-single}"
CAN="${CAN:-can0}"
LEFT_LEADER_CAN="${LEFT_LEADER_CAN:-${LEADER_CAN:-${LEFT_MASTER_CAN:-${CAN}}}}"
RIGHT_LEADER_CAN="${RIGHT_LEADER_CAN:-${RIGHT_MASTER_CAN:-}}"
LEFT_FOLLOWER_CAN="${LEFT_FOLLOWER_CAN:-${FOLLOWER_CAN:-${LEFT_SLAVE_CAN:-${CAN}}}}"
RIGHT_FOLLOWER_CAN="${RIGHT_FOLLOWER_CAN:-${RIGHT_SLAVE_CAN:-}}"
RESTORE_REQUIRE_LEADER="${RESTORE_REQUIRE_LEADER:-1}"
RESTORE_ROLE="${RESTORE_ROLE:-leader}"
RESTORE_ORDER="${RESTORE_ORDER:-leader_first}"
RESTORE_ASSUME_SINGLE_ARM_ON_BUS="${RESTORE_ASSUME_SINGLE_ARM_ON_BUS:-0}"
MASTER_HOME_PREFLIGHT="${MASTER_HOME_PREFLIGHT:-1}"
MASTER_HOME_FIRMWARE_TIMEOUT="${MASTER_HOME_FIRMWARE_TIMEOUT:-3}"
MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE="${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE:-1}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=()

add_single_pair_args() {
  case "${RESTORE_ROLE}" in
    both)
      if [[ -n "${LEFT_LEADER_CAN}" ]]; then
        ARGS+=("--leader-can" "${LEFT_LEADER_CAN}")
      fi
      if [[ -n "${LEFT_FOLLOWER_CAN}" ]]; then
        ARGS+=("--follower-can" "${LEFT_FOLLOWER_CAN}")
      fi
      ;;
    leader)
      if [[ -n "${LEFT_LEADER_CAN}" ]]; then
        ARGS+=("--leader-can" "${LEFT_LEADER_CAN}")
      fi
      ;;
    follower)
      if [[ -n "${LEFT_FOLLOWER_CAN}" ]]; then
        ARGS+=("--follower-can" "${LEFT_FOLLOWER_CAN}")
      fi
      ;;
    *)
      echo "RESTORE_ROLE must be both, leader, or follower, got: ${RESTORE_ROLE}" >&2
      exit 1
      ;;
  esac
}

case "${PAIR_MODE}" in
  single)
    add_single_pair_args
    ;;
  dual)
    if [[ "${RESTORE_ROLE}" == "follower" ]]; then
      echo "RESTORE_ROLE=follower is not implemented for PAIR_MODE=dual in this wrapper." >&2
      exit 1
    fi
    if [[ -n "${LEFT_LEADER_CAN}" ]]; then
      ARGS+=("--leader-can" "${LEFT_LEADER_CAN}")
    fi
    if [[ -n "${RIGHT_LEADER_CAN}" ]]; then
      ARGS+=("--leader-can" "${RIGHT_LEADER_CAN}")
    fi
    if [[ -n "${LEFT_FOLLOWER_CAN}" ]]; then
      ARGS+=("--follower-can" "${LEFT_FOLLOWER_CAN}")
    fi
    if [[ -n "${RIGHT_FOLLOWER_CAN}" ]]; then
      ARGS+=("--follower-can" "${RIGHT_FOLLOWER_CAN}")
    fi
    ;;
  *)
    echo "PAIR_MODE must be single or dual, got: ${PAIR_MODE}" >&2
    exit 1
    ;;
esac

if [[ "${#ARGS[@]}" -eq 0 ]]; then
  echo "No leader/follower CAN ports configured." >&2
  echo "Set LEFT_LEADER_CAN and LEFT_FOLLOWER_CAN, or LEFT_MASTER_CAN and LEFT_SLAVE_CAN." >&2
  exit 1
fi

if [[ -n "${LEFT_LEADER_CAN}" && -n "${LEFT_FOLLOWER_CAN}" && "${LEFT_LEADER_CAN}" == "${LEFT_FOLLOWER_CAN}" ]]; then
  if [[ "${RESTORE_ASSUME_SINGLE_ARM_ON_BUS}" != "1" ]]; then
    SHARED_ARGS=("--can" "${LEFT_LEADER_CAN}" "--mode" "restore")
    if [[ "${MASTER_HOME_PREFLIGHT}" == "1" ]]; then
      SHARED_ARGS+=("--preflight" "--firmware-timeout" "${MASTER_HOME_FIRMWARE_TIMEOUT}")
      if [[ "${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE}" == "1" ]]; then
        SHARED_ARGS+=("--allow-unknown-firmware")
      fi
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      SHARED_ARGS+=("--dry-run")
    fi

    cat >&2 <<EOF
Shared CAN restore on ${LEFT_LEADER_CAN}: using ReqMasterArmMoveToHome(0).

This restores the existing master-slave/teaching mode without sending
MasterSlaveConfig role-assignment frames. It is the safe normal command after a
master_home zero return when both arms are powered on the same bus.

Important: this does not repair already reversed roles. If the follower is
guiding the leader, configure roles with exactly one physical arm powered or
connected at a time:
  RESTORE_ROLE=follower RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1 CAN=${LEFT_FOLLOWER_CAN} bash piper/piper_act_dataset/restore_leader_follower.sh
  RESTORE_ROLE=leader   RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1 CAN=${LEFT_LEADER_CAN} bash piper/piper_act_dataset/restore_leader_follower.sh
EOF

    echo
    echo "Shared CAN restore command:"
    printf '  %q' "${PYTHON}" "piper/piper_act_dataset/request_master_home.py" "${SHARED_ARGS[@]}"
    printf '\n\n'

    "${PYTHON}" piper/piper_act_dataset/request_master_home.py "${SHARED_ARGS[@]}"
    exit $?
  fi

  if [[ "${RESTORE_ROLE}" == "both" ]]; then
    cat >&2 <<EOF
Refusing RESTORE_ROLE=both with RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1.

The isolated-arm recovery path can assign only the one physical Piper arm that
is powered/connected to ${LEFT_LEADER_CAN}. Run follower and leader recovery as
two separate commands, with only the target arm on the CAN bus each time.
EOF
    exit 1
  fi

  cat >&2 <<EOF
Shared CAN isolated-arm restore:
  role=${RESTORE_ROLE}
  can=${LEFT_LEADER_CAN}

This is only safe because RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1 says exactly one
physical Piper arm is powered/connected to ${LEFT_LEADER_CAN}. If both arms are
on the bus, stop now; the command will be broadcast to both arms.
EOF
fi

if [[ "${RESTORE_REQUIRE_LEADER}" == "1" ]]; then
  case "${PAIR_MODE}" in
    single)
      if [[ "${RESTORE_ROLE}" != "follower" && -z "${LEFT_LEADER_CAN}" ]]; then
        echo "Leader CAN port is required to unlock the leader/input arm." >&2
        echo "Your last command only restored the follower (${LEFT_FOLLOWER_CAN:-none})." >&2
        echo "Run with the leader CAN port, for example:" >&2
        echo "  CAN=can0 bash piper/piper_act_dataset/restore_leader_follower.sh" >&2
        echo "If you intentionally want follower-only restore, set RESTORE_REQUIRE_LEADER=0." >&2
        exit 1
      fi
      ;;
    dual)
      if [[ -z "${LEFT_LEADER_CAN}" || -z "${RIGHT_LEADER_CAN}" ]]; then
        echo "PAIR_MODE=dual requires LEFT_LEADER_CAN and RIGHT_LEADER_CAN when RESTORE_REQUIRE_LEADER=1." >&2
        exit 1
      fi
      ;;
  esac
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

ARGS+=("--restore-order" "${RESTORE_ORDER}")

echo "Restore command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/restore_leader_follower.py" "${ARGS[@]}" "$@"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/restore_leader_follower.py "${ARGS[@]}" "$@"
