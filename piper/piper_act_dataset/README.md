# Piper ALOHA/ACT Dataset Recorder

This directory records AgileX Piper slave-arm trajectories into an
ALOHA/ACT-style HDF5 layout for later VLA fine-tuning.

The intended workflow is:

```text
human operates master arm
        ↓
master arm teleoperates slave arm
        ↓
slave arm performs the real task
        ↓
record slave arm trajectory + optional camera images
```

In this workflow, the master arm is only the teleoperation device. The dataset
is centered on the slave arm because the slave arm is the robot that actually
interacts with the scene.

## Project Data Layout

Keep raw recordings and derived formats under this directory:

```text
piper/piper_act_dataset/data/
├── raw/
│   └── <task_name>/
│       └── episode_0/
│           ├── episode_0.hdf5
│           └── episode_0.qa.json
├── readable/
│   └── <task_name>/
│       └── episode_0/
│           ├── metadata.json
│           ├── robot_timeline.csv
│           ├── summary.json
│           └── images/
└── robotwin_pi0/
    └── <task_name>/
        └── episode_0/
            ├── episode_0.hdf5
            └── instructions.json
```

`raw` is the source of truth from collection. `readable` and `robotwin_pi0` are
derived outputs that can be regenerated from kept raw episodes.

## Actual ALOHA Layout

The original ALOHA `record_episodes.py` stores one episode per HDF5 file:

```text
episode_N.hdf5
├── attrs
│   └── sim = False
├── observations/
│   ├── images/
│   │   ├── cam_high
│   │   ├── cam_low
│   │   ├── cam_left_wrist
│   │   └── cam_right_wrist
│   ├── qpos
│   ├── qvel
│   └── effort
└── action
```

For stationary bimanual ALOHA, `qpos`, `qvel`, `effort`, and `action` are
`[T, 14]`. The vector order is:

```text
left  arm: joint_1..joint_6, gripper
right arm: joint_1..joint_6, gripper
```

Images are `uint8` arrays with shape `[T, 480, 640, 3]`, normally one dataset
per camera under `/observations/images/<camera_name>`.

## Piper Mapping

For Piper master-slave teaching:

- `observations/qpos`: slave/output arm feedback from `GetArmJointMsgs()` and
  `GetArmGripperMsgs()`.
- `observations/qvel`: finite difference of `qpos`.
- `observations/effort`: currently zero-filled, because Piper effort/current
  needs a separate calibration decision before it should be used as ALOHA
  effort.
- `action`: by default, the next slave state, i.e. `action[t] = qpos[t+1]`.

Joint angles are stored in radians. Gripper opening is stored in meters.

Why keep `/action` if we only care about the slave trajectory? Many ACT/VLA data
loaders expect an `action` dataset. Since your data source is the slave
trajectory, the most direct supervised label is the next slave state. The model
then learns:

```text
current image + current slave qpos -> next slave qpos
```

The recorder supports three action modes:

```text
slave_next_qpos     default; action[t] = qpos[t+1]
slave_current_qpos  action[t] = qpos[t]
master_ctrl         optional; action comes from master control frames
```

Use `master_ctrl` only if you later decide to train against the master arm's
command stream. It is not the default because your stated goal is to record the
slave arm's real executed trajectory.

The recorder also supports two pair modes:

```text
single  one master-slave pair; stores the active slave arm in the left 7 dims,
        and zero-fills the right 7 dims.
dual    two master-slave pairs; requires left and right slave CAN ports, and
        stores both sides in the 14-D vector.
```

## Dependencies

The local `agilex` conda environment is Python 3.6, so use Python 3.6-compatible
versions:

```bash
clashon
conda activate agilex
pip install numpy==1.19.5 h5py==3.1.0 opencv-python==4.6.0.66
pip install -e /path/to/piper_sdk
```

`opencv-python` is only required when using cameras.

This environment also has a small `sitecustomize.py` compatibility shim so the
local `piper_sdk` can import `typing.Literal` while running on Python 3.6.

## Operator Entry Point

Use `collect_act_episode.sh` for normal repeated collection. It is the single
operator-facing launcher for this directory:

```bash
cd Agilex-College
bash piper/piper_act_dataset/collect_act_episode.sh
```

The shell wrapper calls `collect_episode.py`, which performs preflight checks,
records one or more episodes, runs technical quality analysis, optionally
exports camera previews, and writes a `episode_N.qa.json` keep/reject sidecar
for each episode.

The lower-level recorder, `record_episodes_piper.py`, remains available for
debugging and scripted integrations, but operators should normally use
`collect_act_episode.sh`.

## End-to-End Process

1. Configure the Piper arms.
   - Teaching input arm: `MasterSlaveConfig(0xFA, 0, 0, 0)`.
   - Motion output arm: `MasterSlaveConfig(0xFC, 0, 0, 0)`.
2. Bring up CAN.
   ```bash
   CAN=can0 bash piper/piper_act_dataset/can_up.sh
   ```
3. Verify physical master-slave following before recording.
4. Find the camera device if a USB camera was reconnected.
   ```bash
   bash piper/piper_act_dataset/find_cameras.sh
   ```
5. Record a short pilot episode.
   ```bash
   TASK=press_ring \
   DURATION=5 \
   NO_CAMERA=1 \
   bash piper/piper_act_dataset/collect_act_episode.sh
   ```
6. Inspect the generated HDF5 file.
   ```bash
   python3 piper/piper_act_dataset/inspect_episode.py \
     piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
   ```
7. Add cameras and collect pilot episodes with images.
   ```bash
   TASK=press_ring \
   DURATION=15 \
   CAMERAS="cam_high=/dev/video4" \
   bash piper/piper_act_dataset/collect_act_episode.sh
   ```
8. Review the console quality report, preview images, and `episode_N.qa.json`.
   Keep only episodes with correct task behavior, meaningful arm motion, valid
   timing, and usable images.
9. Repeat collection until the task has enough kept episodes.
10. Export readable or training-specific derived formats only after raw episodes
    are verified.

`raw` episodes are the source of truth. The `readable` and `robotwin_pi0`
outputs are derived files that can be regenerated.

## Repeated Collection Commands

Single master-slave pair, stored in the left 7 dimensions:

```bash
TASK=press_ring \
DURATION=15 \
LEFT_SLAVE_CAN=can0 \
CAMERAS="cam_high=/dev/video4" \
bash piper/piper_act_dataset/collect_act_episode.sh
```

Fast repeated collection without the keep/reject prompt:

```bash
NO_ASK_KEEP=1 \
TASK=press_ring \
DURATION=15 \
LEFT_SLAVE_CAN=can0 \
CAMERAS="cam_high=/dev/video4" \
bash piper/piper_act_dataset/collect_act_episode.sh
```

Collect exactly `episode_0` through `episode_50` in one run:

```bash
TASK=press_ring \
EPISODE_IDX=0 \
NUM_EPISODES=51 \
DURATION=15 \
LEFT_SLAVE_CAN=can0 \
CAMERAS="cam_high=/dev/video4" \
bash piper/piper_act_dataset/collect_act_episode.sh
```

If `EPISODE_IDX` is omitted, each repeat automatically uses the next available
episode index under `data/raw/<task_name>/`.

Two master-slave pairs:

```bash
PAIR_MODE=dual \
TASK=press_ring \
DURATION=15 \
LEFT_SLAVE_CAN=can0 \
RIGHT_SLAVE_CAN=can1 \
CAMERAS="cam_high=/dev/video0 cam_left_wrist=/dev/video2 cam_right_wrist=/dev/video4" \
bash piper/piper_act_dataset/collect_act_episode.sh
```

Dry-run without recording:

```bash
DRY_RUN=1 bash piper/piper_act_dataset/collect_act_episode.sh
```

Disable cameras:

```bash
NO_CAMERA=1 bash piper/piper_act_dataset/collect_act_episode.sh
```

Useful environment variables:

```text
TASK              task subdirectory under data/raw; default press_ring
DURATION          recording length in seconds; default 15
EPISODE_LEN       explicit frame count; overrides DURATION when set
FPS               sampling frequency; default 50
PAIR_MODE         single or dual; default single
ACTION_SOURCE     slave_next_qpos, slave_current_qpos, or master_ctrl
LEFT_SLAVE_CAN    default can0
RIGHT_SLAVE_CAN   required only for PAIR_MODE=dual
CAMERAS           space-separated name=device pairs
NO_CAMERA         set to 1 to collect robot state only
IMAGE_WIDTH       default 320
IMAGE_HEIGHT      default 240
EXPORT_PREVIEW    set to 1 by default
EXPORT_VIDEO      set to 1 to also export MP4 preview
SKIP_PREFLIGHT    set to 1 to skip CAN/camera checks
NO_ASK_KEEP       set to 1 for unattended repeated collection
DRY_RUN           set to 1 to check config without recording
```

## Checking and Exporting Episodes

Inspect HDF5 structure:

```bash
python3 piper/piper_act_dataset/inspect_episode.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
```

Preview the latest episode:

```bash
bash piper/piper_act_dataset/preview_act_episode.sh
```

Export one episode into readable JSON, CSV, and sampled JPG files:

```bash
python3 piper/piper_act_dataset/export_episode_readable.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
```

Convert one verified raw episode to RobotWin Pi0/Pi0.5 format:

```bash
python3 piper/piper_act_dataset/convert_episode_robotwin_pi0.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5 \
  --instruction "press the ring"
```

Replay a recorded trajectory in dry-run mode first:

```bash
DRY_RUN=1 bash piper/piper_act_dataset/replay_act_episode.sh \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
```
