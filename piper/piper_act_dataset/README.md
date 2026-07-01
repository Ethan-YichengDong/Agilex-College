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

## Example Commands

One master-slave pair, stored in the left 7 dimensions:

```bash
cd Agilex-College
python3 piper/piper_act_dataset/record_episodes_piper.py \
  --dataset-dir datasets/piper_act_test \
  --episode-idx 0 \
  --episode-len 1000 \
  --fps 50 \
  --pair-mode single \
  --action-source slave_next_qpos \
  --left-slave-can can0
```

Two master-slave pairs, with cameras:

```bash
cd Agilex-College
python3 piper/piper_act_dataset/record_episodes_piper.py \
  --dataset-dir datasets/piper_act_test \
  --episode-len 1000 \
  --fps 50 \
  --pair-mode dual \
  --action-source slave_next_qpos \
  --left-slave-can can0 \
  --right-slave-can can1 \
  --camera cam_high=/dev/video0 \
  --camera cam_left_wrist=/dev/video2 \
  --camera cam_right_wrist=/dev/video4
```

The master CAN arguments are not needed in the default `slave_next_qpos` mode.
They are only used when `--action-source master_ctrl`.

Inspect an episode:

```bash
python3 piper/piper_act_dataset/inspect_episode.py datasets/piper_act_test/episode_0.hdf5
```

Or use the commented pipeline launcher:

```bash
cd Agilex-College
bash piper/piper_act_dataset/run_piper_act_pipeline.sh
```

Pipeline launcher examples:

```bash
# One master-slave pair
PAIR_MODE=single \
LEFT_SLAVE_CAN=can0 \
bash piper/piper_act_dataset/run_piper_act_pipeline.sh

# Two master-slave pairs
PAIR_MODE=dual \
LEFT_SLAVE_CAN=can0 \
RIGHT_SLAVE_CAN=can1 \
bash piper/piper_act_dataset/run_piper_act_pipeline.sh
```

Dry-run without connecting to CAN:

```bash
DRY_RUN=1 bash piper/piper_act_dataset/run_piper_act_pipeline.sh
```

## Recommended Workflow

1. Configure Piper master/slave mode with `MasterSlaveConfig(0xFA, 0, 0, 0)` for
   the teaching input arm and `MasterSlaveConfig(0xFC, 0, 0, 0)` for the motion
   output arm.
2. Verify physical following without recording.
3. Record one short episode without cameras and inspect the HDF5 file.
4. Add cameras and check image shape/dtype.
5. Collect several pilot episodes and verify the HDF5 file. In the default
   `slave_next_qpos` mode, `action` should look like `qpos` shifted one frame
   forward.
