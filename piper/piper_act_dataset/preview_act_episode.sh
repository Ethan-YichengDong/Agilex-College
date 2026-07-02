#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-datasets/piper_act_collection}"
EPISODE="${1:-}"
CAMERA="${CAMERA:-}"
EXPORT_DIR="${EXPORT_DIR:-}"
EXPORT_VIDEO="${EXPORT_VIDEO:-0}"
JSON="${JSON:-0}"

if [[ -z "${EPISODE}" ]]; then
  EPISODE="$(latest_episode "${DATASET_DIR}")"
fi
if [[ -z "${EPISODE}" ]]; then
  echo "No episode provided and no episode_*.hdf5 found under ${DATASET_DIR}." >&2
  echo "Usage: $0 path/to/episode_N.hdf5" >&2
  exit 1
fi

ARGS=("${EPISODE}")
if [[ -n "${CAMERA}" ]]; then
  ARGS+=("--camera" "${CAMERA}")
fi
if [[ -n "${EXPORT_DIR}" ]]; then
  ARGS+=("--export-dir" "${EXPORT_DIR}")
fi
if [[ "${EXPORT_VIDEO}" == "1" ]]; then
  ARGS+=("--export-video")
fi
if [[ "${JSON}" == "1" ]]; then
  ARGS+=("--json")
fi

"${PYTHON}" piper/piper_act_dataset/preview_episode.py "${ARGS[@]}" "${@:2}"
