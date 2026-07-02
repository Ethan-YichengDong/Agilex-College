#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a Piper ACT HDF5 episode into human-readable files.

The exporter creates:
  metadata.json        root attrs and dataset shapes/dtypes
  summary.json         compact robot/camera statistics
  robot_timeline.csv   one row per timestep with timestamp, qpos, qvel, action
  images/<camera>/     sampled JPG frames, or every frame with --all-images
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


NAMES = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
    "right_gripper",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def infer_task_and_episode(episode: str) -> tuple:
    episode_abs = os.path.abspath(episode)
    episode_name = os.path.splitext(os.path.basename(episode_abs))[0]
    episode_dir = os.path.basename(os.path.dirname(episode_abs))
    task_name = os.path.basename(os.path.dirname(os.path.dirname(episode_abs)))
    if episode_dir == episode_name:
        return task_name, episode_name
    return os.path.basename(os.path.dirname(episode_abs)), episode_name


def default_output_dir(episode: str) -> str:
    episode_abs = os.path.abspath(episode)
    raw_root = os.path.join(DATA_DIR, "raw")
    try:
        common = os.path.commonpath([episode_abs, raw_root])
    except ValueError:
        common = ""
    if common == raw_root:
        task_name, episode_name = infer_task_and_episode(episode_abs)
        return os.path.join(DATA_DIR, "readable", task_name, episode_name)

    episode_base = os.path.splitext(os.path.basename(episode_abs))[0]
    return os.path.join(os.path.dirname(episode_abs), f"{episode_base}_readable")


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def dataset_index(root: h5py.File) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            result[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

    root.visititems(visit)
    return result


def finite_minmax(values: np.ndarray) -> Dict[str, List[float]]:
    return {
        "min": [float(v) for v in np.min(values, axis=0)],
        "max": [float(v) for v in np.max(values, axis=0)],
        "range": [float(v) for v in np.ptp(values, axis=0)],
    }


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_robot_csv(path: str, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray, timestamps_ns: Optional[np.ndarray]) -> None:
    columns = ["timestep"]
    if timestamps_ns is not None:
        columns += ["timestamp_ns", "elapsed_s"]
    columns += [f"qpos_{name}" for name in NAMES]
    columns += [f"qvel_{name}" for name in NAMES]
    columns += [f"action_{name}" for name in NAMES]

    first_timestamp = int(timestamps_ns[0]) if timestamps_ns is not None and len(timestamps_ns) else 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for idx in range(len(qpos)):
            row: List[Any] = [idx]
            if timestamps_ns is not None:
                timestamp = int(timestamps_ns[idx])
                row += [timestamp, (timestamp - first_timestamp) / 1e9]
            row += [float(v) for v in qpos[idx]]
            row += [float(v) for v in qvel[idx]]
            row += [float(v) for v in action[idx]]
            writer.writerow(row)


def frame_indices(length: int, sample_count: int, all_images: bool) -> Iterable[int]:
    if all_images or length <= sample_count:
        return range(length)
    return np.linspace(0, length - 1, sample_count, dtype=int).tolist()


def export_images(root: h5py.File, out_dir: str, sample_count: int, all_images: bool) -> Dict[str, Any]:
    image_group = root.get("observations/images")
    if image_group is None:
        return {"camera_names": [], "exported_frames": {}}
    if cv2 is None:
        raise RuntimeError("opencv-python is required to export images")

    result: Dict[str, Any] = {"camera_names": list(image_group.keys()), "exported_frames": {}}
    image_root = os.path.join(out_dir, "images")
    os.makedirs(image_root, exist_ok=True)

    for camera in image_group.keys():
        images = image_group[camera]
        camera_dir = os.path.join(image_root, camera)
        os.makedirs(camera_dir, exist_ok=True)
        written = []
        for idx in frame_indices(len(images), sample_count, all_images):
            rgb = images[idx]
            filename = f"{camera}_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(camera_dir, filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written.append(filename)
        result["exported_frames"][camera] = {
            "count": len(written),
            "directory": os.path.relpath(camera_dir, out_dir),
            "files": written,
        }
    return result


def export_episode(episode: str, out_dir: str, sample_count: int, all_images: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(episode, "r") as root:
        qpos = root["observations/qpos"][:]
        qvel = root["observations/qvel"][:]
        action = root["action"][:]
        timestamps_ns = root["observations/timestamp_ns"][:] if "observations/timestamp_ns" in root else None

        metadata = {
            "episode": episode,
            "attrs": {key: json_value(value) for key, value in root.attrs.items()},
            "datasets": dataset_index(root),
            "joint_columns": NAMES,
        }
        write_json(os.path.join(out_dir, "metadata.json"), metadata)

        summary: Dict[str, Any] = {
            "episode": episode,
            "timesteps": int(len(qpos)),
            "qpos": finite_minmax(qpos),
            "qvel": finite_minmax(qvel),
            "action": finite_minmax(action),
            "action_shift_max_err": float(np.max(np.abs(action[:-1] - qpos[1:]))) if len(qpos) > 1 else 0.0,
        }
        if timestamps_ns is not None and len(timestamps_ns) > 1:
            dt_ms = np.diff(timestamps_ns) / 1e6
            summary["timing_ms"] = {
                "mean": float(dt_ms.mean()),
                "min": float(dt_ms.min()),
                "max": float(dt_ms.max()),
                "std": float(dt_ms.std()),
            }

        image_summary = export_images(root, out_dir, sample_count, all_images)
        summary.update(image_summary)
        write_json(os.path.join(out_dir, "summary.json"), summary)

        write_robot_csv(os.path.join(out_dir, "robot_timeline.csv"), qpos, qvel, action, timestamps_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="Path to episode_N.hdf5")
    parser.add_argument("--out-dir", default=None, help="Directory to write readable files")
    parser.add_argument("--sample-count", type=int, default=24, help="Number of image frames per camera to export by default")
    parser.add_argument("--all-images", action="store_true", help="Export every camera frame instead of sampled frames")
    args = parser.parse_args()

    out_dir = args.out_dir or default_output_dir(args.episode)
    export_episode(args.episode, out_dir, max(1, args.sample_count), args.all_images)
    print(f"exported readable episode to {out_dir}")


if __name__ == "__main__":
    main()
