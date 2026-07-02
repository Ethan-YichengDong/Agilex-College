#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a Piper ACT episode into the RobotWin OpenPI/Pi0 intermediate HDF5 layout.

RobotWin's Pi0/Pi0.5 flow expects:

training_data/<model_or_group>/<task_name>/episode_0/
├── instructions.json
└── episode_0.hdf5

The HDF5 file keeps ALOHA-style qpos/action arrays and stores images as JPEG
byte strings under /observations/images/{cam_high,cam_left_wrist,cam_right_wrist}.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


ROBOTWIN_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def encode_jpeg_frames(frames_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to encode RobotWin image byte streams")

    encoded: List[bytes] = []
    max_len = 0
    for frame_rgb in frames_rgb:
        resized_rgb = cv2.resize(frame_rgb, (width, height))
        frame_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("failed to JPEG-encode a camera frame")
        payload = jpg.tobytes()
        encoded.append(payload)
        max_len = max(max_len, len(payload))

    return np.array([payload.ljust(max_len, b"\0") for payload in encoded], dtype=f"S{max_len}")


def camera_source_images(image_group: h5py.Group, source_name: str, length: int, width: int, height: int) -> np.ndarray:
    if source_name in image_group:
        return image_group[source_name][:]
    return np.zeros((length, height, width, 3), dtype=np.uint8)


def resolve_camera_map(values: Iterable[str]) -> Dict[str, str]:
    mapping = {name: name for name in ROBOTWIN_CAMERAS}
    for value in values:
        if "=" not in value:
            raise ValueError(f"camera map must be robotwin_name=source_name, got {value!r}")
        target, source = value.split("=", 1)
        if target not in ROBOTWIN_CAMERAS:
            raise ValueError(f"unknown RobotWin camera {target!r}; expected one of {ROBOTWIN_CAMERAS}")
        mapping[target] = source
    return mapping


def infer_task_name(source_episode: str) -> str:
    source_abs = os.path.abspath(source_episode)
    episode_name = os.path.splitext(os.path.basename(source_abs))[0]
    episode_dir = os.path.basename(os.path.dirname(source_abs))
    if episode_dir == episode_name:
        return os.path.basename(os.path.dirname(os.path.dirname(source_abs)))
    return os.path.basename(os.path.dirname(source_abs))


def infer_episode_idx(source_episode: str) -> int:
    episode_name = os.path.splitext(os.path.basename(source_episode))[0]
    prefix = "episode_"
    if episode_name.startswith(prefix):
        return int(episode_name[len(prefix) :])
    raise ValueError(f"cannot infer episode index from {source_episode!r}; pass --episode-idx")


def default_output_dir(source_episode: str, output_root: str, task_name: str) -> str:
    return os.path.join(output_root, task_name)


def convert_episode(
    source_episode: str,
    output_dir: str,
    episode_idx: int,
    instruction: str,
    camera_map: Dict[str, str],
    image_width: int,
    image_height: int,
) -> str:
    episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)

    target_hdf5 = os.path.join(episode_dir, f"episode_{episode_idx}.hdf5")
    instruction_path = os.path.join(episode_dir, "instructions.json")

    with h5py.File(source_episode, "r") as src:
        qpos = src["/observations/qpos"][:]
        action = src["/action"][:]
        qvel = src["/observations/qvel"][:] if "/observations/qvel" in src else None
        effort = src["/observations/effort"][:] if "/observations/effort" in src else None
        image_group = src.get("/observations/images")
        if image_group is None:
            raise RuntimeError("source episode has no /observations/images group")

        # RobotWin's processed Pi0 data uses N-1 rows: qpos[0..N-2] and action
        # as the next state. Piper's default action already stores qpos[t+1].
        qpos_out = qpos[:-1].astype(np.float32)
        action_out = action[:-1].astype(np.float32)

        with h5py.File(target_hdf5, "w") as dst:
            dst.create_dataset("action", data=action_out, dtype="float32")
            obs = dst.create_group("observations")
            obs.create_dataset("qpos", data=qpos_out, dtype="float32")
            if qvel is not None:
                obs.create_dataset("qvel", data=qvel[:-1].astype(np.float32), dtype="float32")
            if effort is not None:
                obs.create_dataset("effort", data=effort[:-1].astype(np.float32), dtype="float32")
            obs.create_dataset("left_arm_dim", data=np.full(len(qpos_out), 6, dtype=np.int64))
            obs.create_dataset("right_arm_dim", data=np.full(len(qpos_out), 6, dtype=np.int64))

            images = obs.create_group("images")
            for robotwin_name in ROBOTWIN_CAMERAS:
                source_name = camera_map[robotwin_name]
                frames = camera_source_images(image_group, source_name, len(qpos), image_width, image_height)[:-1]
                images.create_dataset(
                    robotwin_name,
                    data=encode_jpeg_frames(frames, image_width, image_height),
                )

    with open(instruction_path, "w", encoding="utf-8") as handle:
        json.dump({"instructions": [instruction]}, handle, indent=2)
        handle.write("\n")

    return episode_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_episode", help="Piper ACT episode_N.hdf5")
    parser.add_argument("--output-dir", default=None, help="RobotWin task folder to write episode_N into")
    parser.add_argument("--output-root", default=os.path.join(DATA_DIR, "robotwin_pi0"), help="Root used when --output-dir is omitted")
    parser.add_argument("--task-name", default=None, help="RobotWin task folder name; inferred from raw path when omitted")
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--instruction", default="press the ring", help="Language instruction for instructions.json")
    parser.add_argument(
        "--camera-map",
        action="append",
        default=[],
        help=(
            "Map RobotWin camera name to source camera name, e.g. "
            "cam_high=cam_high. Missing source cameras are written as black frames."
        ),
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    args = parser.parse_args()

    task_name = args.task_name or infer_task_name(args.source_episode)
    episode_idx = args.episode_idx if args.episode_idx is not None else infer_episode_idx(args.source_episode)
    output_dir = args.output_dir or default_output_dir(args.source_episode, args.output_root, task_name)
    episode_dir = convert_episode(
        args.source_episode,
        output_dir,
        episode_idx,
        args.instruction,
        resolve_camera_map(args.camera_map),
        args.image_width,
        args.image_height,
    )
    print(f"converted RobotWin Pi0/Pi0.5 episode to {episode_dir}")


if __name__ == "__main__":
    main()
