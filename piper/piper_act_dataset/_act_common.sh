#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

activate_agilex() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-agilex}"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-agilex}"
  elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV:-agilex}/bin/python" ]]; then
    PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV:-agilex}/bin/python"
    return 0
  else
    echo "Cannot find conda or ${HOME}/miniconda3/envs/${CONDA_ENV:-agilex}/bin/python." >&2
    echo "Install/activate the agilex environment first." >&2
    return 1
  fi
  PYTHON="${PYTHON:-python}"
}

latest_episode() {
  local dataset_dir="${1}"
  find "${dataset_dir}" -name 'episode_*.hdf5' -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

bool_arg() {
  local value="${1}"
  local flag="${2}"
  if [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" ]]; then
    printf '%s\n' "${flag}"
  fi
}
