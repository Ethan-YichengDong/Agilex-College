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
records one or more episodes, runs technical quality analysis, and writes a
`episode_N.qa.json` keep/reject sidecar for each episode. Image viewing is done
later by extracting the raw HDF5 file. Quality checks only print feedback and
data ranges; they do not force rejection.

Before the first episode and after each episode, the default behavior is
`PARK_METHOD=master_home`: the launcher sends the SDK
`ReqMasterArmMoveToHome(2)` request so the leader and follower return to zero
together, waits, then sends `ReqMasterArmMoveToHome(0)` to restore master-slave
mode for the next demonstration. This is the preferred shared-`can0` path
because it does not broadcast new `MasterSlaveConfig` role commands during the
normal collection loop. By default, the script also polls follower feedback and
requires the joints to be near zero before recording starts or continues to the
next trajectory.

The default sampling target remains `FPS=50`. If the console reports about
30Hz, the recorder loop or CAN/SDK feedback path is falling behind the 20ms
target period; saved `qvel` and HDF5 `fps` metadata are computed from actual
timestamps so the mismatch is visible instead of hidden.

When cameras are enabled, the recorder requests `CAMERA_FPS=60` and reads each
camera in a background thread. This prevents a 30Hz camera `read()` call from
blocking the 50Hz robot sampling loop.

`ReqMasterArmMoveToHome` is documented in the SDK as firmware V1.7-4 and later.
On this shared-`can0` setup, the installed SDK reports only a partial firmware
string such as `S-V1`, but hardware testing confirmed that
`ReqMasterArmMoveToHome` works. The operator scripts therefore allow unknown
firmware by default and rely on live zero-pose feedback verification after the
home command.

Do not send `MasterSlaveConfig` role-assignment commands while both arms are
powered/connected on shared `can0`. That command is sent as CAN ID `0x470`; on
the current shared-bus setup it can affect both arms, causing drop or reversed
leader/follower roles. `restore_leader_follower.sh` now detects this shared-bus
case and uses `ReqMasterArmMoveToHome(0)` instead, which restores the existing
master-slave/teaching mode without reassigning roles. To repair already
reversed roles, configure one physical arm at a time.

The lower-level recorder, `record_episodes_piper.py`, remains available for
debugging and scripted integrations, but operators should normally use
`collect_act_episode.sh`.

## End-to-End Process

1. Configure the Piper arms before collection.
   - Teaching input arm: `MasterSlaveConfig(0xFA, 0, 0, 0)`.
   - Motion output arm: `MasterSlaveConfig(0xFC, 0, 0, 0)`.
   - If both arms share `can0`, configure only one powered/connected arm at a
     time; do not send both role commands while both arms are on the bus.
2. Bring up CAN.
   ```bash
   CAN=can0 bash piper/piper_act_dataset/can_up.sh
   ```
3. Verify physical master-slave following before recording.
4. Check that the shared-`can0` home/restore command is supported.
   ```bash
   MASTER_HOME_CHECK=1 bash piper/piper_act_dataset/request_master_home.sh
   ```
   This checks the installed SDK and queries firmware without sending the
   `ReqMasterArmMoveToHome(2)` motion command. Firmware should be `S-V1.7-4` or
   newer for the automatic shared-`can0` reset path.
5. Run the supervised shared-`can0` validation once before official collection.
   ```bash
   bash piper/piper_act_dataset/validate_collection_pipeline.sh
   ```
   This performs the SDK/firmware check, runs zero+restore, pauses for a manual
   leader-follower direction check, verifies that follower feedback moves while
   the leader is moved, resets again, records one short no-camera pilot episode,
   and validates the resulting HDF5 structure.
6. Find the camera device if a USB camera was reconnected.
   ```bash
   bash piper/piper_act_dataset/find_cameras.sh
   ```
   The script classifies readable streams and prints a ready-to-copy
   `CAMERAS="cam_high=... cam_left_wrist=... cam_right_wrist=..."` command.
   For RealSense D435 cameras at `IMAGE_WIDTH=320 IMAGE_HEIGHT=240`, prefer
   streams reported as `RGB candidate`. Skip readable `424`-wide streams,
   because those are usually depth/IR streams rather than RGB.
   A RealSense stable `by-id` path may contain `Depth_Camera_435` because that
   is the product name of the device; it does not by itself identify the stream
   as depth. The finder prefers the color-interface `by-path` name when it can.
7. Record a short pilot episode.
   ```bash
   TASK=press_ring \
   DURATION=5 \
   NO_CAMERA=1 \
   bash piper/piper_act_dataset/collect_act_episode.sh
   ```
   By default, this first runs the `master_home` reset cycle so the pilot starts
   from zero with master-slave mode restored.
8. Inspect the generated HDF5 file.
   ```bash
   python3 piper/piper_act_dataset/inspect_episode.py \
     piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
   ```
9. Add cameras and collect pilot episodes with images.
   ```bash
   TASK=press_ring \
   DURATION=15 \
   NO_CAMERA=0 \
   CAMERA_FPS=60 \
   CAMERAS="cam_high=/dev/video4 cam_left_wrist=/dev/video10 cam_right_wrist=/dev/video16" \
   bash piper/piper_act_dataset/collect_act_episode.sh
   ```
10. Let the script run the `master_home` reset cycle before episode 1 and after
   each episode: `ReqMasterArmMoveToHome(2)` for leader+follower zero return,
   wait, then `ReqMasterArmMoveToHome(0)` to restore master-slave mode. Re-check
   physical following before collecting a large batch.
   If zero verification fails, the script stops before recording the next
   trajectory.
11. Review the console quality report and `episode_N.qa.json`.
   Keep only episodes with correct task behavior, meaningful arm motion, valid
   timing, and usable images.
12. Repeat collection until the task has enough kept episodes.
13. Export readable or training-specific derived formats only after raw episodes
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
LEFT_LEADER_CAN=can2 \
RIGHT_LEADER_CAN=can3 \
CAMERAS="cam_high=/dev/video0 cam_left_wrist=/dev/video2 cam_right_wrist=/dev/video4" \
bash piper/piper_act_dataset/collect_act_episode.sh
```

Dry-run without recording:

```bash
DRY_RUN=1 bash piper/piper_act_dataset/collect_act_episode.sh
```

Dry-run the supervised validation wrapper:

```bash
DRY_RUN=1 bash piper/piper_act_dataset/validate_collection_pipeline.sh
```

Disable cameras:

```bash
NO_CAMERA=1 bash piper/piper_act_dataset/collect_act_episode.sh
```

Disable automatic reset only for debugging:

```bash
PREPARE_BEFORE=0 PARK_AFTER=0 bash piper/piper_act_dataset/collect_act_episode.sh
```

Useful environment variables:

```text
TASK              task subdirectory under data/raw; default test
NUM_EPISODES      number of episodes to collect in this run; default 1
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
CAMERA_FPS        requested camera FPS; default 60
PREPARE_BEFORE    set to 1 by default; reset to zero before episode 1
PARK_AFTER        set to 1 by default; return slave arm(s) to zero after each episode
PARK_METHOD       master_home by default; use can_park only for direct CAN-control debugging
MASTER_HOME_CAN   default can0; CAN bus for ReqMasterArmMoveToHome
MASTER_HOME_WAIT  seconds to wait after ReqMasterArmMoveToHome(2); default 6
MASTER_HOME_RESTORE set to 1 by default; send ReqMasterArmMoveToHome(0) after zero return
MASTER_HOME_PREFLIGHT set to 1 by default; check SDK/firmware before collection
MASTER_HOME_CHECK set to 1 in request_master_home.sh to check SDK/firmware support
MASTER_HOME_FIRMWARE_TIMEOUT firmware query timeout for master_home checks; default 3
MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE set to 1 by default on this setup because
                  the SDK returns partial firmware text but hardware testing passed
MASTER_HOME_VERIFY_ZERO set to 1 by default; poll follower feedback after zero return
MASTER_HOME_ZERO_TIMEOUT seconds to wait for zero verification; default 8
MASTER_HOME_JOINT_TOLERANCE max absolute joint error in radians; default 0.08
MASTER_HOME_VERIFY_GRIPPER set to 1 to also verify gripper position
MASTER_HOME_GRIPPER_TOLERANCE gripper zero tolerance in meters; default 0.01
PARK_SECONDS      zero-return interpolation time; default 5
PARK_MOVE_SPEED   zero-return move speed percentage; default 20
PARK_GRIPPER_EFFORT zero-return gripper effort; default 1000
PARK_NO_GRIPPER   set to 1 to park joints only
PARK_TRY_CAN_MODE set to 1 by default; try switching to CAN mode before parking
PARK_TIMEOUT      zero-return control timeout in seconds; default 5
RESTORE_LEADER_FOLLOWER set to 0 by default; shared-can0 restore is unsafe
RESTORE_ROLE      leader by default; use both only after hardware validation
RESTORE_ASSUME_SINGLE_ARM_ON_BUS set to 1 only when exactly one physical arm is
                  powered/connected to the target CAN bus for role recovery
LEFT_LEADER_CAN   leader/input arm CAN port for restore, e.g. can1
                  defaults to LEFT_SLAVE_CAN for shared-bus single-pair setups
RIGHT_LEADER_CAN  right leader/input arm CAN port for dual mode
LEFT_FOLLOWER_CAN follower/output arm CAN port for restore; defaults to LEFT_SLAVE_CAN
RIGHT_FOLLOWER_CAN right follower/output arm CAN port for dual mode
SKIP_PREFLIGHT    set to 1 to skip CAN/camera checks
NO_ASK_KEEP       set to 1 for unattended repeated collection
DRY_RUN           set to 1 to check config without recording
RUN_PILOT         set to 0 in validate_collection_pipeline.sh to stop before recording
VALIDATION_DURATION pilot duration for validate_collection_pipeline.sh; default 3
VERIFY_LEADER_FOLLOWING set to 1 by default; sample follower feedback while moving leader
LEADER_FOLLOW_VERIFY_DURATION seconds to sample follower feedback; default 4
LEADER_FOLLOW_MIN_RANGE minimum follower joint range in radians; default 0.03
```

Manual shared-can0 zero return and restore for the leader/follower pair:

```bash
MASTER_HOME_CHECK=1 bash piper/piper_act_dataset/request_master_home.sh
```

On this setup, unknown firmware is allowed by default because the SDK returns
partial firmware text while the hardware command has been verified. To force a
strict firmware parse, set `MASTER_HOME_ALLOW_UNKNOWN_FIRMWARE=0`.

```bash
bash piper/piper_act_dataset/request_master_home.sh
```

That runs this cycle:

```text
ReqMasterArmMoveToHome(2)  leader and follower return-to-zero together
wait MASTER_HOME_WAIT
verify follower joint feedback is near zero
ReqMasterArmMoveToHome(0)  restore master-slave mode
```

The zero verifier is enabled by default for the cycle command. Disable it only
for diagnosis:

```bash
MASTER_HOME_VERIFY_ZERO=0 bash piper/piper_act_dataset/request_master_home.sh
```

Send only one SDK master-home request:

```bash
MASTER_HOME_CYCLE=0 MASTER_HOME_MODE=both_zero bash piper/piper_act_dataset/request_master_home.sh
MASTER_HOME_CYCLE=0 MASTER_HOME_MODE=restore bash piper/piper_act_dataset/request_master_home.sh
```

Manual direct CAN-control zero return fallback:

```bash
PARK_METHOD=can_park bash piper/piper_act_dataset/collect_act_episode.sh
```

Use `park_piper_zero.sh` directly only for supervised debugging. It tries to put
the arm into CAN-control mode and can fight the connected master on shared
`can0`.

Manual zero return for a specific CAN port:

```bash
CAN=can1 bash piper/piper_act_dataset/park_piper_zero.sh
```

Manual zero return for two slave arms:

```bash
PAIR_MODE=dual \
LEFT_SLAVE_CAN=can0 \
RIGHT_SLAVE_CAN=can1 \
bash piper/piper_act_dataset/park_piper_zero.sh
```

Manual restore of leader-follower teaching mode on shared `can0`:

```bash
CAN=can0 \
bash piper/piper_act_dataset/restore_leader_follower.sh
```

On shared `can0`, that wrapper sends `ReqMasterArmMoveToHome(0)`. It restores
the existing master-slave relationship after a zero return, but it does not
repair already reversed roles.

If the leader/follower roles are reversed, stop using shared-bus restore and
recover by configuring one physical arm at a time:

```bash
# Only the intended follower/output arm is powered or connected to can0.
RESTORE_ROLE=follower \
RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1 \
CAN=can0 \
bash piper/piper_act_dataset/restore_leader_follower.sh

# Only the intended leader/input arm is powered or connected to can0.
RESTORE_ROLE=leader \
RESTORE_ASSUME_SINGLE_ARM_ON_BUS=1 \
CAN=can0 \
bash piper/piper_act_dataset/restore_leader_follower.sh
```

After that, reconnect both arms and power the follower first, then the leader.
Verify physical leader-to-follower motion before collecting the next episode.

If the leader and follower are on different CAN interfaces, one-command restore
is allowed:

```bash
LEFT_LEADER_CAN=can1 \
LEFT_FOLLOWER_CAN=can0 \
RESTORE_ROLE=both \
bash piper/piper_act_dataset/restore_leader_follower.sh
```

## Checking and Exporting Episodes

Inspect HDF5 structure:

```bash
python3 piper/piper_act_dataset/inspect_episode.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
```

Strictly validate Piper ACT HDF5 structure:

```bash
python3 piper/piper_act_dataset/inspect_episode.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5 \
  --validate-act \
  --expected-camera cam_high \
  --expected-camera cam_left_wrist \
  --expected-camera cam_right_wrist \
  --require-images
```

Preview the latest episode:

```bash
bash piper/piper_act_dataset/preview_act_episode.sh
```

Export one episode into readable JSON, CSV, and full JPG image frames:

```bash
python3 piper/piper_act_dataset/export_episode_readable.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5
```

By default this writes every saved camera timestep under `readable/images`.
For a small preview export only, use `--sample-images`:

```bash
python3 piper/piper_act_dataset/export_episode_readable.py \
  piper/piper_act_dataset/data/raw/press_ring/episode_0/episode_0.hdf5 \
  --sample-images
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
