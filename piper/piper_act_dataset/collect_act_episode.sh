#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_act_common.sh
source "${SCRIPT_DIR}/_act_common.sh"

activate_agilex
cd "${REPO_ROOT}"

# =========================
# 采集参数
# =========================

# 原始 HDF5 数据保存根目录。
# 实际保存路径会是：
#   ${DATASET_DIR}/${TASK}/episode_N/episode_N.hdf5
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/data/raw}"

# 任务名称，也就是 data/raw 下的一级子目录名。
# 例如 TASK=press_ring 会保存到：
#   piper/piper_act_dataset/data/raw/press_ring/
TASK="${TASK:-test}"

# 起始 episode 编号。
# 留空时自动选择下一个可用编号；例如已有 episode_0，则自动从 episode_1 开始。
# 设置 EPISODE_IDX=0 且 NUM_EPISODES=51 时，会采集 episode_0 到 episode_50。
EPISODE_IDX="${EPISODE_IDX:-}"

# 本次连续采集的 episode 数量。
# 每个 episode 都会单独提示按 Enter 开始，并单独保存为一个 HDF5 文件。
NUM_EPISODES="${NUM_EPISODES:-1}"

# 每个 episode 的采集时长，单位：秒。
# 只有在 EPISODE_LEN 为空时生效；最终帧数约为 DURATION * FPS。
DURATION="${DURATION:-15}"

# 每个 episode 的固定帧数。
# 留空时由 DURATION 和 FPS 自动计算；设置后会覆盖 DURATION。
EPISODE_LEN="${EPISODE_LEN:-}"

# 采样频率，单位：Hz。
# ACT/ALOHA 常用 50Hz；例如 DURATION=15、FPS=50 时约采集 750 帧。
FPS="${FPS:-50}"

# 机械臂对数模式。
# single：单臂采集，只使用 left 7 维，right 7 维补 0。
# dual：双臂采集，需要同时设置 LEFT_SLAVE_CAN 和 RIGHT_SLAVE_CAN。
PAIR_MODE="${PAIR_MODE:-single}"

# /action 字段来源。
# slave_next_qpos：默认值，action[t] = qpos[t+1]，适合轨迹模仿学习。
# slave_current_qpos：action[t] = qpos[t]。
# master_ctrl：从主臂控制帧读取 action，仅在明确需要主臂指令流时使用。
ACTION_SOURCE="${ACTION_SOURCE:-slave_next_qpos}"

# 左侧/单臂从臂 CAN 口，用来读取真实执行机械臂反馈。
# single 模式下必须设置，一般是 can0。
LEFT_SLAVE_CAN="${LEFT_SLAVE_CAN:-can0}"

# 右侧从臂 CAN 口。
# 仅 PAIR_MODE=dual 时使用。
RIGHT_SLAVE_CAN="${RIGHT_SLAVE_CAN:-}"

# 左侧主臂 CAN 口。
# 仅 ACTION_SOURCE=master_ctrl 时使用；普通 slave_next_qpos 模式不需要。
LEFT_MASTER_CAN="${LEFT_MASTER_CAN:-}"

# 右侧主臂 CAN 口。
# 仅 ACTION_SOURCE=master_ctrl 且双臂时使用。
RIGHT_MASTER_CAN="${RIGHT_MASTER_CAN:-}"

# 单摄像头快捷配置的摄像头名称。
# 建议使用 RobotWin 兼容名称：cam_high、cam_left_wrist、cam_right_wrist。
CAMERA_NAME="${CAMERA_NAME:-cam_high}"

# 单摄像头快捷配置的设备节点。
# 可先运行 find_cameras.sh 查找可用 /dev/videoX。
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video4}"

# 摄像头列表，格式为：
#   "相机名=设备节点 相机名=设备节点"
# 例如：
#   CAMERAS="cam_high=/dev/video0 cam_left_wrist=/dev/video2 cam_right_wrist=/dev/video4"
# 如果没有手动设置 CAMERAS，则默认使用 CAMERA_NAME=CAMERA_DEVICE 这一组。
CAMERAS="${CAMERAS:-${CAMERA_NAME}=${CAMERA_DEVICE}}"

# 是否禁用摄像头。
# 1：不采集图像，只采集机械臂状态。
# 0：使用 CAMERAS 中配置的摄像头。
NO_CAMERA="${NO_CAMERA:-1}"

# 保存到 HDF5 的图像宽度，单位：像素。
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"

# 保存到 HDF5 的图像高度，单位：像素。
IMAGE_HEIGHT="${IMAGE_HEIGHT:-240}"

# 是否导出预览图。
# 1：采集完成后导出少量 JPG 预览到 data/readable。
# 0：不导出预览。
EXPORT_PREVIEW="${EXPORT_PREVIEW:-1}"

# 是否额外导出 MP4 预览视频。
# 只有 EXPORT_PREVIEW=1 时才有意义。
EXPORT_VIDEO="${EXPORT_VIDEO:-0}"

# 是否跳过采集前检查。
# 0：检查 CAN 口和摄像头是否可用。
# 1：跳过检查，适合你已经确认硬件状态时快速采集。
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

# 是否跳过每个 episode 结束后的 keep/reject 人工确认。
# 0：每条轨迹结束后询问是否保留。
# 1：不询问，只写自动 QA 结果。
NO_ASK_KEEP="${NO_ASK_KEEP:-0}"

# 只打印最终命令和解析后的配置，不真正采集。
# 用于检查参数是否正确。
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
  "--dataset-dir" "${DATASET_DIR}"
  "--num-episodes" "${NUM_EPISODES}"
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
if [[ -n "${EPISODE_IDX}" ]]; then
  ARGS+=("--episode-idx" "${EPISODE_IDX}")
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
