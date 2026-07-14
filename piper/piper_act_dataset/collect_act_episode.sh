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
TASK="${TASK:-click_silver_bell_left}"

# 起始 episode 编号。
# 留空时自动选择下一个可用编号；例如已有 episode_0，则自动从 episode_1 开始。
# 设置 EPISODE_IDX=0 且 NUM_EPISODES=51 时，会采集 episode_0 到 episode_50。
EPISODE_IDX="${EPISODE_IDX:-}"

# 本次连续采集的 episode 数量。
# 每个 episode 都会单独提示按 Enter 开始，并单独保存为一个 HDF5 文件。
NUM_EPISODES="${NUM_EPISODES:-30}"

# 每个 episode 的采集时长，单位：秒。
# 只有在 EPISODE_LEN 为空时生效；最终帧数约为 DURATION * FPS。
DURATION="${DURATION:-15}"

# 每个 episode 的固定帧数。
# 留空时由 DURATION 和 FPS 自动计算；设置后会覆盖 DURATION。
EPISODE_LEN="${EPISODE_LEN:-400}"

# 采样频率，单位：Hz。
# 目标是 50Hz；如果实际记录速度低于 50Hz，说明采样循环或 CAN/SDK 反馈链路跟不上。
FPS="${FPS:-50}"

# 机械臂对数模式。
# single：单臂采集，只采集一对主从臂；具体写入 left 7 维还是 right 7 维
#         由 SINGLE_ARM_SIDE 控制，另一侧补 0。
# dual：双臂采集，需要同时设置 LEFT_SLAVE_CAN 和 RIGHT_SLAVE_CAN。
PAIR_MODE="${PAIR_MODE:-single}"

# single 模式下写入 ACT 14 维向量的哪一侧。
# left：默认，兼容已有数据集。
# right：单右臂采集，left 7 维补 0，right 7 维保存真实数据。
SINGLE_ARM_SIDE="${SINGLE_ARM_SIDE:-left}"

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

# 回零后恢复主从/leader-follower 模式时使用的 leader CAN 口。
# 如果和 LEFT_MASTER_CAN / RIGHT_MASTER_CAN 是同一个含义，可以只设置 MASTER_CAN。
LEFT_LEADER_CAN="${LEFT_LEADER_CAN:-${LEFT_MASTER_CAN:-${LEFT_SLAVE_CAN}}}"
RIGHT_LEADER_CAN="${RIGHT_LEADER_CAN:-${RIGHT_MASTER_CAN}}"

# 回零后恢复主从/leader-follower 模式时使用的 follower CAN 口。
# 默认等于从臂 CAN 口。
LEFT_FOLLOWER_CAN="${LEFT_FOLLOWER_CAN:-${LEFT_SLAVE_CAN}}"
RIGHT_FOLLOWER_CAN="${RIGHT_FOLLOWER_CAN:-${RIGHT_SLAVE_CAN}}"

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
CAMERAS="cam_high=/dev/video10 cam_left_wrist=/dev/video4 cam_right_wrist=/dev/video16"

# 是否禁用摄像头。
# 1：不采集图像，只采集机械臂状态。
# 0：使用 CAMERAS 中配置的摄像头。
NO_CAMERA="${NO_CAMERA:-0}"

# 保存到 HDF5 的图像宽度，单位：像素。
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"

# 保存到 HDF5 的图像高度，单位：像素。
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"

# 请求摄像头帧率，单位：Hz。你的相机支持 30/60Hz 时，默认请求 60Hz，
# 避免 30Hz 摄像头阻塞 50Hz 机械臂采样循环。
CAMERA_FPS="${CAMERA_FPS:-60}"

# 是否跳过采集前检查。
# 0：检查 CAN 口和摄像头是否可用。
# 1：跳过检查，适合你已经确认硬件状态时快速采集。
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

# 是否跳过每个 episode 结束后的 keep/reject 人工确认。
# 0：每条轨迹结束后询问是否保留。
# 1：不询问，只写自动 QA 结果。
NO_ASK_KEEP="${NO_ASK_KEEP:-0}"

# 正式开始第一条轨迹前是否先执行一次回零+恢复主从模式。
# 1：默认开启，保证本次采集的第一条轨迹也从同一个零点开始。
# 0：关闭，仅用于你已经手动确认两臂在起点且主从关系正常时。
PREPARE_BEFORE="${PREPARE_BEFORE:-1}"

# 每条轨迹结束后是否自动把从臂回到零点。
# 1：默认开启，保证下一条轨迹从相同起点开始。
# 0：关闭，仅用于调试或人工确认硬件不允许自动回零时。
PARK_AFTER="${PARK_AFTER:-1}"

# 回零方式。
# master_home：默认，使用 SDK ReqMasterArmMoveToHome(2) 让主从臂一起回零，
#              等待后再用 ReqMasterArmMoveToHome(0) 恢复主从模式。
# can_park：旧路径，强制从臂进入 CAN 控制后插值到零点；共享 can0 时不建议。
PARK_METHOD="${PARK_METHOD:-master_home}"

# master_home 回零使用的 CAN 口。你的 leader/follower 共用 can0 时保持默认即可。
MASTER_HOME_CAN="${MASTER_HOME_CAN:-${CAN:-${LEFT_SLAVE_CAN}}}"

# 发送 ReqMasterArmMoveToHome(2) 后等待多久再恢复主从模式。
MASTER_HOME_WAIT="${MASTER_HOME_WAIT:-5}"

# master_home 回零后是否发送 ReqMasterArmMoveToHome(0) 恢复主从模式。
MASTER_HOME_RESTORE="${MASTER_HOME_RESTORE:-1}"

# 采集前是否检查 SDK/固件是否支持 ReqMasterArmMoveToHome。
# 只查询固件，不发送回零运动指令；SKIP_PREFLIGHT=1 会跳过。
MASTER_HOME_PREFLIGHT="${MASTER_HOME_PREFLIGHT:-1}"

# master_home 支持检查中的固件查询超时。
MASTER_HOME_FIRMWARE_TIMEOUT="${MASTER_HOME_FIRMWARE_TIMEOUT:-3}"

# 是否允许固件版本无法解析时继续采集。
# 你的共享 can0 硬件测试中 SDK 只返回 S-V1，但 ReqMasterArmMoveToHome 可正常工作；
# 因此默认允许未知固件，并继续依赖实际回零反馈确认来保护采集流程。
MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE="${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE:-1}"

# master_home 回零后是否读取反馈并确认从臂关节已经接近零点。
# 1：默认开启；确认失败会停止采集，避免下一条轨迹起点漂移。
MASTER_HOME_VERIFY_ZERO="${MASTER_HOME_VERIFY_ZERO:-1}"

# 回零确认最多等待多久，单位：秒。
MASTER_HOME_ZERO_TIMEOUT="${MASTER_HOME_ZERO_TIMEOUT:-8}"

# 关节回零容差，单位：rad。0.08 rad 约等于 4.6 度。
MASTER_HOME_JOINT_TOLERANCE="${MASTER_HOME_JOINT_TOLERANCE:-0.08}"

# 是否同时确认夹爪也接近 0m。默认关闭，避免 home 命令不控制夹爪时误拦截。
MASTER_HOME_VERIFY_GRIPPER="${MASTER_HOME_VERIFY_GRIPPER:-0}"

# 夹爪回零容差，单位：m；只有 MASTER_HOME_VERIFY_GRIPPER=1 时使用。
MASTER_HOME_GRIPPER_TOLERANCE="${MASTER_HOME_GRIPPER_TOLERANCE:-0.01}"

# 回零插值时间，单位：秒。时间越长，动作越慢。
PARK_SECONDS="${PARK_SECONDS:-5}"

# 回零速度百分比，传给 Piper move speed。
PARK_MOVE_SPEED="${PARK_MOVE_SPEED:-20}"

# 回零时夹爪控制力度。
PARK_GRIPPER_EFFORT="${PARK_GRIPPER_EFFORT:-1000}"

# 回零时是否不控制夹爪。
PARK_NO_GRIPPER="${PARK_NO_GRIPPER:-0}"

# 回零时尝试切换从臂到 CAN 控制模式。
# 采集通常处于主从/示教状态；默认开启，确保脚本能发回零指令。
PARK_TRY_CAN_MODE="${PARK_TRY_CAN_MODE:-1}"

# 回零控制超时时间，单位：秒。
PARK_TIMEOUT="${PARK_TIMEOUT:-5}"

# 回零后是否自动恢复 leader-follower 主从模式。
# 共享 can0 的硬件测试表明 MasterSlaveConfig 可能同时影响两台机械臂，
# 导致掉电或主从关系反转。因此默认关闭。只有在你确认该命令能正确
# 定向到单台机械臂时才手动设置为 1。
RESTORE_LEADER_FOLLOWER="${RESTORE_LEADER_FOLLOWER:-0}"

# 回零后恢复哪个角色。
# leader：默认，只释放 leader/input arm，保持已有 follower 配置。
# both：同时发送 leader 0xFA 和 follower 0xFC，仅在确认硬件需要时使用。
# follower：只发送 follower 0xFC，通常不建议用于共享 can0。
RESTORE_ROLE="${RESTORE_ROLE:-leader}"

# 只打印最终命令和解析后的配置，不真正采集。
# 用于检查参数是否正确。
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
  "--dataset-dir" "${DATASET_DIR}"
  "--num-episodes" "${NUM_EPISODES}"
  "--duration" "${DURATION}"
  "--fps" "${FPS}"
  "--pair-mode" "${PAIR_MODE}"
  "--single-arm-side" "${SINGLE_ARM_SIDE}"
  "--action-source" "${ACTION_SOURCE}"
  "--left-slave-can" "${LEFT_SLAVE_CAN}"
  "--image-width" "${IMAGE_WIDTH}"
  "--image-height" "${IMAGE_HEIGHT}"
  "--camera-fps" "${CAMERA_FPS}"
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
if [[ "${SKIP_PREFLIGHT}" == "1" ]]; then
  ARGS+=("--skip-preflight")
fi
if [[ "${NO_ASK_KEEP}" == "1" ]]; then
  ARGS+=("--no-ask-keep")
fi
if [[ "${PREPARE_BEFORE}" == "1" ]]; then
  ARGS+=("--prepare-before")
fi
if [[ "${PARK_AFTER}" == "1" ]]; then
  ARGS+=("--park-after")
fi
if [[ "${PREPARE_BEFORE}" == "1" || "${PARK_AFTER}" == "1" ]]; then
  ARGS+=("--park-method" "${PARK_METHOD}")
  ARGS+=("--master-home-can" "${MASTER_HOME_CAN}")
  ARGS+=("--master-home-wait" "${MASTER_HOME_WAIT}")
  if [[ "${MASTER_HOME_PREFLIGHT}" == "1" ]]; then
    ARGS+=("--master-home-preflight")
  fi
  ARGS+=("--master-home-firmware-timeout" "${MASTER_HOME_FIRMWARE_TIMEOUT}")
  if [[ "${MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE}" == "1" ]]; then
    ARGS+=("--master-home-allow-unknown-firmware")
  fi
  if [[ "${MASTER_HOME_RESTORE}" != "1" ]]; then
    ARGS+=("--no-master-home-restore")
  fi
  if [[ "${MASTER_HOME_VERIFY_ZERO}" == "1" ]]; then
    ARGS+=("--master-home-verify-zero")
    ARGS+=("--master-home-zero-timeout" "${MASTER_HOME_ZERO_TIMEOUT}")
    ARGS+=("--master-home-joint-tolerance" "${MASTER_HOME_JOINT_TOLERANCE}")
    if [[ "${MASTER_HOME_VERIFY_GRIPPER}" == "1" ]]; then
      ARGS+=("--master-home-gripper-tolerance" "${MASTER_HOME_GRIPPER_TOLERANCE}")
    fi
  fi
  ARGS+=("--park-seconds" "${PARK_SECONDS}")
  ARGS+=("--park-move-speed" "${PARK_MOVE_SPEED}")
  ARGS+=("--park-gripper-effort" "${PARK_GRIPPER_EFFORT}")
  ARGS+=("--park-timeout" "${PARK_TIMEOUT}")
  if [[ "${PARK_NO_GRIPPER}" == "1" ]]; then
    ARGS+=("--park-no-gripper")
  fi
  if [[ "${PARK_TRY_CAN_MODE}" == "1" ]]; then
    ARGS+=("--park-try-can-mode")
  fi
  if [[ "${RESTORE_LEADER_FOLLOWER}" == "1" ]]; then
    ARGS+=("--restore-leader-follower")
    ARGS+=("--restore-role" "${RESTORE_ROLE}")
    if [[ -n "${LEFT_LEADER_CAN}" ]]; then
      ARGS+=("--left-leader-can" "${LEFT_LEADER_CAN}")
    fi
    if [[ -n "${RIGHT_LEADER_CAN}" ]]; then
      ARGS+=("--right-leader-can" "${RIGHT_LEADER_CAN}")
    fi
    if [[ -n "${LEFT_FOLLOWER_CAN}" ]]; then
      ARGS+=("--left-follower-can" "${LEFT_FOLLOWER_CAN}")
    fi
    if [[ -n "${RIGHT_FOLLOWER_CAN}" ]]; then
      ARGS+=("--right-follower-can" "${RIGHT_FOLLOWER_CAN}")
    fi
  fi
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

echo "Collect command:"
printf '  %q' "${PYTHON}" "piper/piper_act_dataset/collect_episode.py" "${ARGS[@]}" "$@"
printf '\n\n'

"${PYTHON}" piper/piper_act_dataset/collect_episode.py "${ARGS[@]}" "$@"
