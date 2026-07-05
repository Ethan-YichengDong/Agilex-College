#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record Piper slave-arm demonstrations in an ALOHA/ACT-compatible HDF5 layout.

Each episode is saved as:

episode_N.hdf5
├── observations/
│   ├── images/<camera_name>        optional, uint8, [T, H, W, 3], RGB
│   ├── qpos                        float32, [T, 14]
│   ├── qvel                        float32, [T, 14]
│   └── effort                      float32, [T, 14], zero-filled by default
└── action                          float32, [T, 14]

The 14-D vector follows ALOHA's bimanual convention:
left_arm_joint_1..6, left_gripper, right_arm_joint_1..6, right_gripper.

This recorder is designed for your workflow:

1. A human operates the master arm.
2. The master arm teleoperates the slave arm.
3. The dataset records the slave arm's real executed trajectory.

By default, `/observations/qpos` is the slave state at time t, and `/action` is
the slave state at time t+1. This means the training sample is:

current image + current slave state -> next slave state.

That default matches a trajectory-only VLA fine-tuning dataset. Reading master
control frames is still available as an optional mode, but it is not the default.

Use `--pair-mode single` for one master-slave pair. The active pair is stored in
the left 7 dimensions and the right 7 dimensions are zero-filled. Use
`--pair-mode dual` for two master-slave pairs; both left and right slave CAN
ports are required.
"""

import argparse
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from piper_sdk import C_PiperInterface_V2

try:
    import cv2
except ImportError:  # Cameras are optional.
    cv2 = None


RAD_PER_MILLI_DEG = np.pi / 180000.0


class CameraSpec:
    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.device = device


class CameraReader:
    def __init__(self, specs: Iterable[CameraSpec], width: int, height: int, fps: float) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required when --camera is used")
        self._captures: Dict[str, "cv2.VideoCapture"] = {}
        self._threads: List[threading.Thread] = []
        self._frames: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = True
        self._width = width
        self._height = height
        self._fps = fps
        for spec in specs:
            cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"failed to open camera {spec.name}: {spec.device}")
            self._captures[spec.name] = cap
            thread = threading.Thread(target=self._reader_loop, args=(spec.name, cap), daemon=True)
            thread.start()
            self._threads.append(thread)

        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._lock:
                ready = all(name in self._frames for name in self._captures)
            if ready:
                break
            time.sleep(0.02)
        with self._lock:
            missing = [name for name in self._captures if name not in self._frames]
        if missing:
            self.close()
            raise RuntimeError(f"failed to read initial frame from camera(s): {missing}")

        for name, cap in self._captures.items():
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {name}: requested_fps={fps:.1f}, driver_fps={actual_fps:.1f}")

    def _reader_loop(self, name: str, cap: "cv2.VideoCapture") -> None:
        while self._running:
            ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frames[name] = frame
            else:
                time.sleep(0.005)

    @property
    def names(self) -> List[str]:
        return list(self._captures.keys())

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self._height, self._width, 3

    def read(self) -> Dict[str, np.ndarray]:
        images = {}
        with self._lock:
            frames = {name: frame.copy() for name, frame in self._frames.items()}
        for name in self._captures:
            frame_bgr = frames.get(name)
            if frame_bgr is None:
                raise RuntimeError(f"no frame available from camera {name}")
            frame_bgr = cv2.resize(frame_bgr, (self._width, self._height))
            images[name] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return images

    def close(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=0.5)
        for cap in self._captures.values():
            cap.release()


def connect_piper(can_name: str) -> C_PiperInterface_V2:
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.1)
    return piper


def joint_feedback_rad(piper: C_PiperInterface_V2) -> np.ndarray:
    msg = piper.GetArmJointMsgs().joint_state
    return np.array([getattr(msg, f"joint_{i}") for i in range(1, 7)], dtype=np.float32) * RAD_PER_MILLI_DEG


def joint_control_rad(piper: C_PiperInterface_V2) -> np.ndarray:
    msg = piper.GetArmJointCtrl()
    return np.array([getattr(msg, f"joint_{i}") for i in range(1, 7)], dtype=np.float32) * RAD_PER_MILLI_DEG


def gripper_feedback_m(piper: C_PiperInterface_V2) -> float:
    return float(piper.GetArmGripperMsgs().gripper_state.grippers_angle) / 1e6


def gripper_control_m(piper: C_PiperInterface_V2) -> float:
    # Piper reports the gripper control frame as grippers_angle. In the regular
    # control API this is sent in micrometers; keep meters in the ACT vector.
    return float(piper.GetArmGripperCtrl().grippers_angle) / 1e6


def side_vector(joints: np.ndarray, gripper: float) -> np.ndarray:
    return np.concatenate([joints, np.array([gripper], dtype=np.float32)]).astype(np.float32)


def bimanual_vector(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
) -> np.ndarray:
    zeros = np.zeros(7, dtype=np.float32)
    return np.concatenate([left if left is not None else zeros, right if right is not None else zeros]).astype(np.float32)


def episode_file_path(dataset_dir: str, episode_idx: int) -> str:
    episode_name = f"episode_{episode_idx}"
    return os.path.join(dataset_dir, episode_name, f"{episode_name}.hdf5")


def next_episode_index(dataset_dir: str) -> int:
    os.makedirs(dataset_dir, exist_ok=True)
    for idx in range(100000):
        if not os.path.exists(episode_file_path(dataset_dir, idx)):
            return idx
    raise RuntimeError("too many episodes in dataset directory")


def parse_camera_specs(values: List[str]) -> List[CameraSpec]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"camera spec must be name=device, got {value!r}")
        name, device = value.split("=", 1)
        if not name or not device:
            raise ValueError(f"camera spec must be name=device, got {value!r}")
        specs.append(CameraSpec(name=name, device=device))
    return specs


def save_episode(
    path: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    effort: np.ndarray,
    timestamps_ns: np.ndarray,
    images: Dict[str, np.ndarray],
    fps: float,
    target_fps: float,
    action_source: str,
    pair_mode: str,
) -> None:
    with h5py.File(path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["fps"] = fps
        root.attrs["target_fps"] = target_fps
        root.attrs["robot"] = "agilex_piper"
        root.attrs["state_dim"] = 14
        root.attrs["vector_order"] = "left_j1..j6,left_gripper,right_j1..j6,right_gripper"
        root.attrs["action_source"] = action_source
        root.attrs["pair_mode"] = pair_mode

        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos, dtype="float32")
        obs.create_dataset("qvel", data=qvel, dtype="float32")
        obs.create_dataset("effort", data=effort, dtype="float32")
        obs.create_dataset("timestamp_ns", data=timestamps_ns, dtype="int64")
        root.create_dataset("action", data=action, dtype="float32")

        if images:
            image_group = obs.create_group("images")
            for name, array in images.items():
                image_group.create_dataset(
                    name,
                    data=array,
                    dtype="uint8",
                    chunks=(1,) + array.shape[1:],
                )


def finite_difference_qvel(qpos: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    qvel = np.zeros_like(qpos, dtype=np.float32)
    if len(qpos) > 1:
        dt = np.diff(timestamps_ns.astype(np.float64)) / 1e9
        dt = np.maximum(dt, 1e-6)
        qvel[1:] = (qpos[1:] - qpos[:-1]) / dt[:, None]
        qvel[0] = qvel[1]
    return qvel


def measured_fps(timestamps_ns: np.ndarray, fallback_fps: float) -> float:
    if len(timestamps_ns) < 2:
        return fallback_fps
    elapsed = (float(timestamps_ns[-1]) - float(timestamps_ns[0])) / 1e9
    if elapsed <= 0:
        return fallback_fps
    return float((len(timestamps_ns) - 1) / elapsed)


def build_action_from_slave_qpos(qpos: np.ndarray, action_source: str) -> np.ndarray:
    if action_source == "slave_current_qpos":
        return qpos.copy()
    if action_source == "slave_next_qpos":
        action = np.empty_like(qpos, dtype=np.float32)
        if len(qpos) > 1:
            action[:-1] = qpos[1:]
            action[-1] = qpos[-1]
        else:
            action[:] = qpos
        return action
    raise ValueError(f"unsupported slave action source: {action_source}")


def capture_episode(args: argparse.Namespace) -> str:
    left_slave = connect_piper(args.left_slave_can) if args.left_slave_can else None
    right_slave = connect_piper(args.right_slave_can) if args.right_slave_can else None
    if args.action_source == "master_ctrl":
        left_master = connect_piper(args.left_master_can) if args.left_master_can else left_slave
        right_master = connect_piper(args.right_master_can) if args.right_master_can else right_slave
    else:
        left_master = None
        right_master = None

    camera_specs = parse_camera_specs(args.camera)
    camera_reader = CameraReader(camera_specs, args.image_width, args.image_height, args.camera_fps) if camera_specs else None

    episode_idx = args.episode_idx if args.episode_idx is not None else next_episode_index(args.dataset_dir)
    episode_path = episode_file_path(args.dataset_dir, episode_idx)
    if os.path.exists(episode_path) and not args.overwrite:
        raise FileExistsError(f"{episode_path} already exists; pass --overwrite to replace it")

    dt = 1.0 / args.fps
    qpos_rows: List[np.ndarray] = []
    master_action_rows: List[np.ndarray] = []
    effort_rows: List[np.ndarray] = []
    timestamp_rows: List[int] = []
    image_rows: Dict[str, List[np.ndarray]] = {name: [] for name in (camera_reader.names if camera_reader else [])}

    input("Press Enter to start recording this episode.")
    start = time.monotonic()
    next_tick = start

    try:
        for step in range(args.episode_len):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            sample_time_ns = time.monotonic_ns()

            left_qpos = side_vector(joint_feedback_rad(left_slave), gripper_feedback_m(left_slave)) if left_slave else None
            right_qpos = side_vector(joint_feedback_rad(right_slave), gripper_feedback_m(right_slave)) if right_slave else None

            qpos_rows.append(bimanual_vector(left_qpos, right_qpos))
            if args.action_source == "master_ctrl":
                left_action = side_vector(joint_control_rad(left_master), gripper_control_m(left_master)) if left_master else None
                right_action = side_vector(joint_control_rad(right_master), gripper_control_m(right_master)) if right_master else None
                master_action_rows.append(bimanual_vector(left_action, right_action))
            effort_rows.append(np.zeros(14, dtype=np.float32))
            timestamp_rows.append(sample_time_ns)

            if camera_reader:
                for name, image in camera_reader.read().items():
                    image_rows[name].append(image)

            next_tick += dt
            if args.print_every and (step + 1) % args.print_every == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                print(f"recorded {step + 1}/{args.episode_len} frames, avg fps={((step + 1) / elapsed):.2f}")
    finally:
        if camera_reader:
            camera_reader.close()

    qpos = np.stack(qpos_rows).astype(np.float32)
    if args.action_source == "master_ctrl":
        action = np.stack(master_action_rows).astype(np.float32)
    else:
        action = build_action_from_slave_qpos(qpos, args.action_source)
    effort = np.stack(effort_rows).astype(np.float32)
    timestamps_ns = np.array(timestamp_rows, dtype=np.int64)
    qvel = finite_difference_qvel(qpos, timestamps_ns)
    actual_fps = measured_fps(timestamps_ns, args.fps)
    if actual_fps < args.fps * 0.9:
        print(
            f"WARNING: requested fps={args.fps:.2f}, measured fps={actual_fps:.2f}. "
            "The HDF5 qvel and fps metadata use measured timestamps."
        )
    image_arrays = {name: np.stack(rows).astype(np.uint8) for name, rows in image_rows.items()}

    os.makedirs(os.path.dirname(episode_path), exist_ok=True)
    save_episode(
        episode_path,
        qpos,
        qvel,
        action,
        effort,
        timestamps_ns,
        image_arrays,
        actual_fps,
        args.fps,
        args.action_source,
        args.pair_mode,
    )
    return episode_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Task directory; episodes are saved as episode_N/episode_N.hdf5")
    parser.add_argument("--episode-idx", type=int, default=None, help="Episode index; auto-selected if omitted")
    parser.add_argument("--episode-len", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--fps", type=float, default=50.0, help="Sampling frequency; ALOHA commonly uses 50Hz")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing episode file")
    parser.add_argument(
        "--pair-mode",
        choices=["single", "dual"],
        default="single",
        help=(
            "single records one master-slave pair into the left 7 dimensions "
            "and zero-fills the right side. dual records two master-slave pairs "
            "and requires both --left-slave-can and --right-slave-can."
        ),
    )
    parser.add_argument(
        "--action-source",
        choices=["slave_next_qpos", "slave_current_qpos", "master_ctrl"],
        default="slave_next_qpos",
        help=(
            "How to populate /action. slave_next_qpos records the next slave state "
            "and is the default for trajectory-only VLA fine-tuning. "
            "slave_current_qpos records the current slave state. master_ctrl reads "
            "master control frames; master CAN args are only needed if those frames "
            "are not visible on the slave CAN interface."
        ),
    )

    parser.add_argument("--left-slave-can", default=None, help="CAN port for left slave/output arm feedback")
    parser.add_argument("--right-slave-can", default=None, help="CAN port for right slave/output arm feedback")
    parser.add_argument("--left-master-can", default=None, help="CAN port carrying left master/input control frames; only used with --action-source master_ctrl")
    parser.add_argument("--right-master-can", default=None, help="CAN port carrying right master/input control frames; only used with --action-source master_ctrl")

    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Optional camera as name=device, e.g. cam_high=/dev/video0. May be repeated.",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=60.0, help="Requested V4L2 camera FPS")
    parser.add_argument("--print-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.pair_mode == "single":
        if not args.left_slave_can:
            raise SystemExit("--pair-mode single requires --left-slave-can.")
        if args.right_slave_can:
            raise SystemExit("--pair-mode single should not set --right-slave-can. Use --pair-mode dual for two pairs.")
    if args.pair_mode == "dual":
        if not args.left_slave_can or not args.right_slave_can:
            raise SystemExit("--pair-mode dual requires both --left-slave-can and --right-slave-can.")
    path = capture_episode(args)
    print(f"Saved episode to {path}")


if __name__ == "__main__":
    main()
