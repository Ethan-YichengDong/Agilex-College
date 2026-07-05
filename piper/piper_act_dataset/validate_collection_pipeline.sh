#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

# Supervised hardware validation for the shared-can0 ACT collection loop.
# This script intentionally pauses for operator confirmation around real arm
# motion and the manual leader-follower check.

CAN="${CAN:-can0}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/data/raw}"
TASK="${TASK:-pipeline_validation}"
VALIDATION_DURATION="${VALIDATION_DURATION:-3}"
FPS="${FPS:-50}"
RUN_PILOT="${RUN_PILOT:-1}"
INSPECT_PILOT="${INSPECT_PILOT:-1}"
VERIFY_LEADER_FOLLOWING="${VERIFY_LEADER_FOLLOWING:-1}"
LEADER_FOLLOW_VERIFY_DURATION="${LEADER_FOLLOW_VERIFY_DURATION:-4}"
LEADER_FOLLOW_SAMPLE_RATE="${LEADER_FOLLOW_SAMPLE_RATE:-25}"
LEADER_FOLLOW_MIN_RANGE="${LEADER_FOLLOW_MIN_RANGE:-0.03}"
AUTO_CONFIRM="${AUTO_CONFIRM:-0}"
DRY_RUN="${DRY_RUN:-0}"
REPORT_DIR="${REPORT_DIR:-${DATASET_DIR}/${TASK}}"
VALIDATION_REPORT="${VALIDATION_REPORT:-}"
CURRENT_STAGE="startup"
episode_path=""
if [[ -z "${SKIP_PREFLIGHT+x}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    SKIP_PREFLIGHT=1
  else
    SKIP_PREFLIGHT=0
  fi
fi

# Keep the validation pilot quick and focused on robot state unless the
# operator explicitly enables cameras through NO_CAMERA=0 and CAMERAS=...
NO_CAMERA="${NO_CAMERA:-1}"
NO_ASK_KEEP="${NO_ASK_KEEP:-1}"

step() {
  CURRENT_STAGE="$1"
  printf '\n== %s ==\n' "$1"
}

report_path() {
  mkdir -p "${REPORT_DIR}"
  if [[ -z "${VALIDATION_REPORT}" ]]; then
    VALIDATION_REPORT="${REPORT_DIR}/validation_$(date +%Y%m%d_%H%M%S).json"
  fi
}

write_report() {
  local status="$1"
  local failed_stage="${2:-}"
  report_path
  REPORT_STATUS="${status}" \
  FAILED_STAGE="${failed_stage}" \
  VALIDATION_REPORT="${VALIDATION_REPORT}" \
  CAN="${CAN}" \
  DATASET_DIR="${DATASET_DIR}" \
  TASK="${TASK}" \
  VALIDATION_DURATION="${VALIDATION_DURATION}" \
  FPS="${FPS}" \
  RUN_PILOT="${RUN_PILOT}" \
  INSPECT_PILOT="${INSPECT_PILOT}" \
  VERIFY_LEADER_FOLLOWING="${VERIFY_LEADER_FOLLOWING}" \
  LEADER_FOLLOW_VERIFY_DURATION="${LEADER_FOLLOW_VERIFY_DURATION}" \
  LEADER_FOLLOW_SAMPLE_RATE="${LEADER_FOLLOW_SAMPLE_RATE}" \
  LEADER_FOLLOW_MIN_RANGE="${LEADER_FOLLOW_MIN_RANGE}" \
  NO_CAMERA="${NO_CAMERA}" \
  CAMERAS="${CAMERAS:-}" \
  SKIP_PREFLIGHT="${SKIP_PREFLIGHT}" \
  DRY_RUN="${DRY_RUN}" \
  EPISODE_PATH="${episode_path}" \
  "${PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone

path = os.environ["VALIDATION_REPORT"]
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": os.environ["REPORT_STATUS"],
    "failed_stage": os.environ["FAILED_STAGE"] or None,
    "can": os.environ["CAN"],
    "dataset_dir": os.environ["DATASET_DIR"],
    "task": os.environ["TASK"],
    "episode_path": os.environ["EPISODE_PATH"] or None,
    "dry_run": os.environ["DRY_RUN"] == "1",
    "skip_preflight": os.environ["SKIP_PREFLIGHT"] == "1",
    "run_pilot": os.environ["RUN_PILOT"] == "1",
    "inspect_pilot": os.environ["INSPECT_PILOT"] == "1",
    "no_camera": os.environ["NO_CAMERA"] == "1",
    "cameras": os.environ["CAMERAS"],
    "validation_duration": float(os.environ["VALIDATION_DURATION"]),
    "fps": float(os.environ["FPS"]),
    "verify_leader_following": os.environ["VERIFY_LEADER_FOLLOWING"] == "1",
    "leader_follow_verify_duration": float(os.environ["LEADER_FOLLOW_VERIFY_DURATION"]),
    "leader_follow_sample_rate": float(os.environ["LEADER_FOLLOW_SAMPLE_RATE"]),
    "leader_follow_min_range": float(os.environ["LEADER_FOLLOW_MIN_RANGE"]),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(f"Validation report written: {path}")
PY
}

on_exit() {
  local code=$?
  if [[ "${code}" -ne 0 ]]; then
    write_report "failed" "${CURRENT_STAGE}"
  fi
}
trap on_exit EXIT

confirm() {
  local prompt="$1"
  if [[ "${DRY_RUN}" == "1" || "${AUTO_CONFIRM}" == "1" ]]; then
    echo "${prompt} [auto-confirmed]"
    return 0
  fi
  local answer
  read -r -p "${prompt} [y/N]: " answer
  case "${answer}" in
    y|Y|yes|YES)
      return 0
      ;;
    *)
      echo "Validation stopped by operator."
      exit 1
      ;;
  esac
}

step "Safety confirmation"
confirm "Arms are clear, emergency stop is reachable, and both arms are powered on ${CAN}"

step "SDK/firmware support check"
CAN="${CAN}" \
MASTER_HOME_CHECK=1 \
DRY_RUN="${DRY_RUN}" \
bash piper/piper_act_dataset/request_master_home.sh

step "Zero return and restore master-slave mode"
CAN="${CAN}" \
DRY_RUN="${DRY_RUN}" \
bash piper/piper_act_dataset/request_master_home.sh

step "Manual leader-follower check"
if [[ "${VERIFY_LEADER_FOLLOWING}" == "1" ]]; then
  VERIFY_ARGS=(
    "--can" "${CAN}"
    "--duration" "${LEADER_FOLLOW_VERIFY_DURATION}"
    "--sample-rate" "${LEADER_FOLLOW_SAMPLE_RATE}"
    "--min-joint-range" "${LEADER_FOLLOW_MIN_RANGE}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    VERIFY_ARGS+=("--dry-run")
  fi
  "${PYTHON}" piper/piper_act_dataset/verify_leader_following.py "${VERIFY_ARGS[@]}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry-run: skip visual leader-follower direction confirmation"
else
  echo "Gently move the leader arm a small amount."
  echo "The follower should track the leader in the correct direction."
  echo "Then bring the leader close to zero or hold it steady; the next step will run zero return again."
  confirm "Did the follower track the leader correctly"
fi

step "Reset again before pilot recording"
CAN="${CAN}" \
DRY_RUN="${DRY_RUN}" \
bash piper/piper_act_dataset/request_master_home.sh

if [[ "${RUN_PILOT}" != "1" ]]; then
  echo "RUN_PILOT=0; validation stopped before pilot recording."
  write_report "stopped_before_pilot" ""
  exit 0
fi

step "Record one short pilot episode"
TASK="${TASK}" \
DATASET_DIR="${DATASET_DIR}" \
DURATION="${VALIDATION_DURATION}" \
FPS="${FPS}" \
NUM_EPISODES=1 \
LEFT_SLAVE_CAN="${CAN}" \
MASTER_HOME_CAN="${CAN}" \
NO_CAMERA="${NO_CAMERA}" \
NO_ASK_KEEP="${NO_ASK_KEEP}" \
PREPARE_BEFORE=0 \
PARK_AFTER=1 \
SKIP_PREFLIGHT="${SKIP_PREFLIGHT}" \
DRY_RUN="${DRY_RUN}" \
bash piper/piper_act_dataset/collect_act_episode.sh

if [[ "${DRY_RUN}" == "1" || "${INSPECT_PILOT}" != "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    write_report "dry_run_passed" ""
  else
    write_report "passed_without_inspection" ""
  fi
  exit 0
fi

step "Inspect latest pilot HDF5"
episode_path="$(latest_episode "${DATASET_DIR}/${TASK}")"
if [[ -z "${episode_path}" ]]; then
  echo "Could not find a pilot episode under ${DATASET_DIR}/${TASK}" >&2
  exit 1
fi

echo "Latest pilot episode: ${episode_path}"
INSPECT_ARGS=("${episode_path}" "--validate-act")
expected_len="$("${PYTHON}" - <<PY
print(int(round(float("${VALIDATION_DURATION}") * float("${FPS}"))))
PY
)"
INSPECT_ARGS+=("--expected-len" "${expected_len}")
if [[ "${NO_CAMERA}" != "1" ]]; then
  INSPECT_ARGS+=("--require-images")
  if [[ -n "${CAMERAS:-}" ]]; then
    for camera in ${CAMERAS}; do
      INSPECT_ARGS+=("--expected-camera" "${camera%%=*}")
    done
  fi
fi

"${PYTHON}" piper/piper_act_dataset/inspect_episode.py "${INSPECT_ARGS[@]}"
write_report "passed" ""
