#!/usr/bin/env bash
#
# Piper 主从示教 -> ALOHA/ACT HDF5 数据采集启动脚本
# ====================================================
#
# 这个脚本是“启动入口 + 操作说明 + pipeline 讲解”。
# 它不直接实现底层采集逻辑；底层采集逻辑在同目录的
# `record_episodes_piper.py` 中。这样做的原因是：
#   1. Bash 负责环境、参数、启动流程；
#   2. Python 负责 CAN 读取、相机读取、HDF5 写入。
#
# 一、这套流程的核心思想
# ----------------------
# 你的需求是：
#
#   人工操作主臂 -> 主臂遥操作从臂 -> 从臂完成任务 -> 记录从臂真实轨迹
#
# 在这套工作流里，主臂不是训练数据的主体；主臂只是一个“遥操作输入设备”。
# 训练数据的主体是从臂，因为从臂才是真正接触环境、完成任务的机械臂。
#
# 对 VLA/ACT 类模型来说，HDF5 中仍然常见 `/action` 字段。这里的 action
# 不默认使用主臂控制帧，而是默认从“从臂轨迹”本身构造：
#
#   observations/qpos[t] = 从臂在第 t 帧的真实状态
#   action[t]            = 从臂在第 t+1 帧的真实状态
#
# 也就是：
#
#   当前图像 + 当前从臂状态 -> 下一步从臂状态
#
# 这和你说的“记录从臂运动轨迹作为数据集，用于后训练 VLA 模型”是一致的。
# 如果以后你明确想复刻 ALOHA 原始主从训练定义，也可以把 ACTION_SOURCE
# 改成 master_ctrl，显式记录主臂控制帧；但默认不是这个模式。
#
# 二、HDF5 文件结构
# -----------------
# 每次示教保存为一个 episode_N.hdf5：
#
#   episode_N.hdf5
#   ├── observations/
#   │   ├── images/<camera_name>    可选，uint8，[T,H,W,3]，RGB
#   │   ├── qpos                    float32，[T,14]
#   │   ├── qvel                    float32，[T,14]
#   │   ├── effort                  float32，[T,14]，当前填 0
#   │   └── timestamp_ns            int64，[T]
#   └── action                      float32，[T,14]，默认是下一帧从臂状态
#
# 14 维顺序固定为：
#
#   left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper
#
# 如果你现在只有一套主从臂，可以只填左臂 7 维，右臂 7 维自动填 0。
#
# 三、Piper SDK 字段映射
# ---------------------
#   qpos:
#     从臂反馈 GetArmJointMsgs() + GetArmGripperMsgs()
#
#   action:
#     默认 ACTION_SOURCE=slave_next_qpos：
#       action[t] = qpos[t+1]，最后一帧 action 复制最后一帧 qpos
#     可选 ACTION_SOURCE=slave_current_qpos：
#       action[t] = qpos[t]
#     可选 ACTION_SOURCE=master_ctrl：
#       action 来自主臂控制帧 GetArmJointCtrl() + GetArmGripperCtrl()
#
#   qvel:
#     对 qpos 按采样周期做有限差分
#
#   effort:
#     当前填 0。后续如果要使用电流/力矩，需要先确认 Piper SDK 对应
#     反馈字段的单位和与 ACT effort 的语义是否一致。
#
# 四、硬件操作 pipeline
# --------------------
#   1. 连接主臂、从臂、CAN 模块和相机。
#   2. 启动 CAN：
#        sudo ip link set can0 up type can bitrate 1000000
#      如果有多个 CAN 口，分别启动 can0/can1/...
#   3. 确认主从模式已经设置：
#        主臂：MasterSlaveConfig(0xFA, 0, 0, 0)
#        从臂：MasterSlaveConfig(0xFC, 0, 0, 0)
#   4. 先不采集，手动验证主臂拖动时从臂能稳定跟随。
#   5. 运行本脚本，按回车开始录制一个 episode。
#   6. 用 inspect_episode.py 检查 HDF5 shape/dtype。
#   7. 采几条短 episode，检查 qpos 和 action 是否符合预期：
#        默认 slave_next_qpos 模式下，action 曲线应该基本是 qpos 曲线向前平移一帧。
#      确认数据正常后再大规模采集。
#
# 五、网络和依赖说明
# -----------------
# 如果你在新终端安装依赖，先执行：
#
#   clashon
#   conda activate agilex
#   pip install numpy==1.19.5 h5py==3.1.0 opencv-python==4.6.0.66
#
# 当前 agilex 环境是 Python 3.6，所以这里使用的是兼容 Python 3.6 的版本。
#
# 六、使用方式
# ------------
# 先进入 Agilex-College 仓库根目录，然后运行：
#
#   bash piper/piper_act_dataset/run_piper_act_pipeline.sh
#
# 方式 B：通过环境变量覆盖配置，例如：
#
#   DATASET_DIR=datasets/test \
#   EPISODE_LEN=500 \
#   PAIR_MODE=single \
#   LEFT_SLAVE_CAN=can0 \
#   bash piper/piper_act_dataset/run_piper_act_pipeline.sh
#
# 双套主从示例：
#
#   PAIR_MODE=dual \
#   LEFT_SLAVE_CAN=can0 \
#   RIGHT_SLAVE_CAN=can1 \
#   bash piper/piper_act_dataset/run_piper_act_pipeline.sh
#
# 方式 C：增加相机。CAMERAS 采用 name=device，用空格分隔：
#
#   CAMERAS="cam_high=/dev/video0 cam_left_wrist=/dev/video2" \
#   bash piper/piper_act_dataset/run_piper_act_pipeline.sh
#
# 方式 D：只检查环境和配置，不连接机械臂：
#
#   DRY_RUN=1 bash piper/piper_act_dataset/run_piper_act_pipeline.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ "$(pwd -P)" == "${REPO_ROOT}" ]]; then
  RECORDER="piper/piper_act_dataset/record_episodes_piper.py"
  INSPECTOR="piper/piper_act_dataset/inspect_episode.py"
else
  RECORDER="${SCRIPT_DIR}/record_episodes_piper.py"
  INSPECTOR="${SCRIPT_DIR}/inspect_episode.py"
fi

# =========================
# 你最常修改的是这些参数
# =========================

# 数据保存目录。每次采集会生成 episode_N.hdf5。
# 默认保存在 Agilex-College/datasets/piper_act。
# 推荐在 Agilex-College 根目录运行本脚本；这样默认值和你传入的
# DATASET_DIR=datasets/test 都是相对于仓库根目录的路径。
DATASET_DIR="${DATASET_DIR:-datasets/piper_act}"

# episode 编号。留空则自动选择下一个可用编号。
EPISODE_IDX="${EPISODE_IDX:-}"

# 采样长度和频率。ALOHA 常用 50Hz；1000 帧约等于 20 秒。
EPISODE_LEN="${EPISODE_LEN:-1000}"
FPS="${FPS:-50}"

# action 字段来源。
# 默认 slave_next_qpos：用从臂下一帧状态作为动作标签，适合“只用从臂轨迹后训练 VLA”。
# 其他可选值：
#   slave_current_qpos：action 等于当前从臂状态。
#   master_ctrl：action 来自主臂控制帧，仅在你明确需要 ALOHA 原始主从定义时使用。
ACTION_SOURCE="${ACTION_SOURCE:-slave_next_qpos}"

# 主从套数。
# single：一套主从，只采 left 7 维，right 7 维补 0。默认模式。
# dual：两套主从，同时采 left/right 两侧，要求 LEFT_SLAVE_CAN 和 RIGHT_SLAVE_CAN 都有效。
PAIR_MODE="${PAIR_MODE:-single}"

# CAN 口配置。
# 单套主从臂：通常只需要 LEFT_SLAVE_CAN=can0。
# 默认不需要主臂 CAN，因为默认只记录从臂轨迹。
# 只有 ACTION_SOURCE=master_ctrl 时，LEFT_MASTER_CAN / RIGHT_MASTER_CAN 才会被使用。
# 如果主臂控制帧和从臂反馈在同一个总线上，MASTER_CAN 可以留空。
# 如果你有双侧双臂，设置 RIGHT_SLAVE_CAN。
LEFT_SLAVE_CAN="${LEFT_SLAVE_CAN:-can0}"
RIGHT_SLAVE_CAN="${RIGHT_SLAVE_CAN:-}"
LEFT_MASTER_CAN="${LEFT_MASTER_CAN:-}"
RIGHT_MASTER_CAN="${RIGHT_MASTER_CAN:-}"

# 相机配置，格式为 "cam_high=/dev/video0 cam_left_wrist=/dev/video2"。
# 不接相机时保持为空，脚本仍会采集 qpos/qvel/action。
CAMERAS="${CAMERAS:-}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"

# 是否覆盖同名 episode。0 表示不覆盖，1 表示覆盖。
OVERWRITE="${OVERWRITE:-0}"

# 只做环境检查和参数展示，不启动采集。
DRY_RUN="${DRY_RUN:-1}"

print_config() {
  cat <<EOF

Piper ALOHA/ACT pipeline config
-------------------------------
DATASET_DIR      = ${DATASET_DIR}
EPISODE_IDX      = ${EPISODE_IDX:-auto}
EPISODE_LEN      = ${EPISODE_LEN}
FPS              = ${FPS}
ACTION_SOURCE    = ${ACTION_SOURCE}
PAIR_MODE        = ${PAIR_MODE}
LEFT_SLAVE_CAN   = ${LEFT_SLAVE_CAN:-none}
RIGHT_SLAVE_CAN  = ${RIGHT_SLAVE_CAN:-none}
LEFT_MASTER_CAN  = ${LEFT_MASTER_CAN:-unused unless ACTION_SOURCE=master_ctrl}
RIGHT_MASTER_CAN = ${RIGHT_MASTER_CAN:-unused unless ACTION_SOURCE=master_ctrl}
CAMERAS          = ${CAMERAS:-none}
IMAGE_SIZE       = ${IMAGE_WIDTH}x${IMAGE_HEIGHT}
OVERWRITE        = ${OVERWRITE}
DRY_RUN          = ${DRY_RUN}

EOF
}

activate_agilex() {
  # conda 在非交互脚本里不一定自动可用。优先使用 PATH 中的 conda；
  # 如果不可用，再尝试常见的 $HOME/miniconda3 位置，避免写死用户名或父目录。
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "Cannot find conda. Please install conda or activate the agilex environment before running." >&2
    return 1
  fi
  conda activate agilex
}

check_dependencies() {
  python - <<'PY'
import importlib

required = ["numpy", "h5py", "cv2", "piper_sdk"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)

if missing:
    raise SystemExit(
        "Missing dependencies in agilex env: {}\n"
        "Install with:\n"
        "  clashon\n"
        "  conda activate agilex\n"
        "  pip install numpy==1.19.5 h5py==3.1.0 opencv-python==4.6.0.66".format(
            ", ".join(missing)
        )
    )

print("Dependency check passed: numpy, h5py, cv2, piper_sdk")
PY
}

validate_pair_mode() {
  case "${PAIR_MODE}" in
    single)
      if [[ -z "${LEFT_SLAVE_CAN}" ]]; then
        echo "PAIR_MODE=single requires LEFT_SLAVE_CAN." >&2
        return 1
      fi
      if [[ -n "${RIGHT_SLAVE_CAN}" ]]; then
        echo "PAIR_MODE=single should not set RIGHT_SLAVE_CAN. Use PAIR_MODE=dual for two pairs." >&2
        return 1
      fi
      ;;
    dual)
      if [[ -z "${LEFT_SLAVE_CAN}" || -z "${RIGHT_SLAVE_CAN}" ]]; then
        echo "PAIR_MODE=dual requires both LEFT_SLAVE_CAN and RIGHT_SLAVE_CAN." >&2
        return 1
      fi
      ;;
    *)
      echo "PAIR_MODE must be single or dual, got: ${PAIR_MODE}" >&2
      return 1
      ;;
  esac
}

build_recorder_args() {
  RECORDER_ARGS=(
    "--dataset-dir" "${DATASET_DIR}"
    "--episode-len" "${EPISODE_LEN}"
    "--fps" "${FPS}"
    "--pair-mode" "${PAIR_MODE}"
    "--action-source" "${ACTION_SOURCE}"
    "--image-width" "${IMAGE_WIDTH}"
    "--image-height" "${IMAGE_HEIGHT}"
  )

  if [[ -n "${EPISODE_IDX}" ]]; then
    RECORDER_ARGS+=("--episode-idx" "${EPISODE_IDX}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    RECORDER_ARGS+=("--overwrite")
  fi
  if [[ -n "${LEFT_SLAVE_CAN}" ]]; then
    RECORDER_ARGS+=("--left-slave-can" "${LEFT_SLAVE_CAN}")
  fi
  if [[ "${PAIR_MODE}" == "dual" && -n "${RIGHT_SLAVE_CAN}" ]]; then
    RECORDER_ARGS+=("--right-slave-can" "${RIGHT_SLAVE_CAN}")
  fi
  if [[ -n "${LEFT_MASTER_CAN}" ]]; then
    RECORDER_ARGS+=("--left-master-can" "${LEFT_MASTER_CAN}")
  fi
  if [[ -n "${RIGHT_MASTER_CAN}" ]]; then
    RECORDER_ARGS+=("--right-master-can" "${RIGHT_MASTER_CAN}")
  fi
  if [[ -n "${CAMERAS}" ]]; then
    for camera in ${CAMERAS}; do
      RECORDER_ARGS+=("--camera" "${camera}")
    done
  fi
}

main() {
  activate_agilex
  check_dependencies
  validate_pair_mode
  print_config
  build_recorder_args

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, recorder command would be:"
    printf '  python %q' "${RECORDER}"
    printf ' %q' "${RECORDER_ARGS[@]}"
    printf '\n'
    return 0
  fi

  echo "About to start recording."
  echo "Make sure the Piper master-slave following is already working physically."
  echo

  python "${RECORDER}" "${RECORDER_ARGS[@]}"

  echo
  echo "Latest files in ${DATASET_DIR}:"
  ls -lh "${DATASET_DIR}" | tail -10
  echo
  echo "To inspect one episode:"
  echo "  conda activate agilex"
  echo "  python ${INSPECTOR} ${DATASET_DIR}/episode_<N>.hdf5"
}

main "$@"
