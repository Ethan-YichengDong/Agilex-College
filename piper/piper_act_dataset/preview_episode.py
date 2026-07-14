#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a Piper ACT HDF5 episode and optionally export quick camera previews.

This is a technical quality check, not a semantic task-success judge. It checks
dataset shape, robot/gripper ranges, timing, action shift consistency, and image
basic statistics so bad captures are easy to reject early.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import h5py
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


SIDE_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def _as_float_list(values: np.ndarray) -> list:
    return [float(v) for v in values]


def _image_stats(images: np.ndarray) -> Dict[str, Any]:
    first = images[0]
    last = images[-1]
    return {
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "min": int(images.min()),
        "max": int(images.max()),
        "mean": float(images.mean()),
        "first_min": int(first.min()),
        "first_max": int(first.max()),
        "first_mean": float(first.mean()),
        "last_min": int(last.min()),
        "last_max": int(last.max()),
        "last_mean": float(last.mean()),
        "first_last_mean_abs_delta": float(np.mean(np.abs(last.astype(np.int16) - first.astype(np.int16)))),
    }


def analyze_episode(path: str, camera: Optional[str] = None) -> Dict[str, Any]:
    with h5py.File(path, "r") as root:
        qpos = root["observations/qpos"][:]
        qvel = root["observations/qvel"][:]
        action = root["action"][:]
        timestamps_ns = root["observations/timestamp_ns"][:] if "observations/timestamp_ns" in root else None
        attrs = {key: _json_value(value) for key, value in root.attrs.items()}
        pair_mode = attrs.get("pair_mode", "single")
        active_side = attrs.get("single_arm_side", "left") if pair_mode == "single" else "left"

        left = qpos[:, :7]
        right = qpos[:, 7:]
        left_ranges = np.ptp(left, axis=0)
        right_ranges = np.ptp(right, axis=0)
        active_ranges = right_ranges if active_side == "right" else left_ranges
        result: Dict[str, Any] = {
            "path": path,
            "attrs": attrs,
            "active_side": active_side,
            "qpos_shape": list(qpos.shape),
            "qvel_shape": list(qvel.shape),
            "action_shape": list(action.shape),
            "qpos_dtype": str(qpos.dtype),
            "action_dtype": str(action.dtype),
            "left_min": _as_float_list(left.min(axis=0)),
            "left_max": _as_float_list(left.max(axis=0)),
            "left_range": _as_float_list(left_ranges),
            "left_qvel_max_abs": _as_float_list(np.max(np.abs(qvel[:, :7]), axis=0)),
            "right_min": _as_float_list(right.min(axis=0)),
            "right_max": _as_float_list(right.max(axis=0)),
            "right_range": _as_float_list(right_ranges),
            "right_qvel_max_abs": _as_float_list(np.max(np.abs(qvel[:, 7:]), axis=0)),
            "active_range": _as_float_list(active_ranges),
            "inactive_side_max_abs": float(
                np.max(np.abs(qpos[:, 7:])) if active_side == "left" else np.max(np.abs(qpos[:, :7]))
            ),
            "action_shift_max_err": float(np.max(np.abs(action[:-1] - qpos[1:]))) if len(qpos) > 1 else 0.0,
        }

        if timestamps_ns is not None and len(timestamps_ns) > 1:
            dt_ms = np.diff(timestamps_ns) / 1e6
            result.update(
                {
                    "dt_ms_mean": float(dt_ms.mean()),
                    "dt_ms_min": float(dt_ms.min()),
                    "dt_ms_max": float(dt_ms.max()),
                    "dt_ms_std": float(dt_ms.std()),
                }
            )

        image_group = root.get("observations/images")
        if image_group is not None:
            camera_names = list(image_group.keys())
            result["camera_names"] = camera_names
            selected = camera or (camera_names[0] if camera_names else None)
            if selected:
                if selected not in image_group:
                    raise KeyError(f"camera {selected!r} not found; available: {camera_names}")
                result["selected_camera"] = selected
                result["image"] = _image_stats(image_group[selected][:])
        else:
            result["camera_names"] = []

    result["quality_flags"] = quality_flags(result)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def quality_flags(result: Dict[str, Any]) -> Dict[str, bool]:
    ranges = np.array(result["active_range"], dtype=np.float32)
    flags = {
        "has_robot_rows": result["qpos_shape"][0] > 0 and result["qpos_shape"][1] == 14,
        "action_is_next_qpos": result["action_shift_max_err"] < 1e-6,
        "single_inactive_side_zero": result["inactive_side_max_abs"] < 1e-6,
        "some_arm_motion": bool(np.any(ranges[:6] > 1e-3)),
        "gripper_motion": bool(ranges[6] > 1e-4),
    }
    if "dt_ms_mean" in result:
        flags["timing_near_50hz"] = 15.0 <= result["dt_ms_mean"] <= 25.0
        flags["no_large_timing_spike"] = result["dt_ms_max"] < 100.0
    if "image" in result:
        image = result["image"]
        flags["image_nonblack"] = image["mean"] > 5.0 and image["max"] > 20
        flags["first_frame_nonblack"] = image["first_mean"] > 5.0 and image["first_max"] > 20
    return flags


def print_report(result: Dict[str, Any]) -> None:
    active_side = result.get("active_side", "left")
    print(f"episode: {result['path']}")
    print(f"qpos: shape={tuple(result['qpos_shape'])}, dtype={result['qpos_dtype']}")
    print(f"action: shape={tuple(result['action_shape'])}, dtype={result['action_dtype']}")
    if "dt_ms_mean" in result:
        print(
            "timing ms: "
            f"mean={result['dt_ms_mean']:.3f}, min={result['dt_ms_min']:.3f}, "
            f"max={result['dt_ms_max']:.3f}, std={result['dt_ms_std']:.3f}"
        )
    print(f"active side: {active_side}")
    print(f"inactive side max abs: {result['inactive_side_max_abs']:.6f}")
    print(f"action shift max err: {result['action_shift_max_err']:.9f}")
    print(f"{active_side} ranges:")
    for name, value in zip(SIDE_NAMES, result[f"{active_side}_range"]):
        print(f"  {name}: {value:.6f}")
    if "image" in result:
        image = result["image"]
        print(
            f"camera {result['selected_camera']}: shape={tuple(image['shape'])}, "
            f"dtype={image['dtype']}, mean={image['mean']:.2f}, "
            f"first_mean={image['first_mean']:.2f}, max={image['max']}"
        )
    print("quality flags:")
    for key, value in result["quality_flags"].items():
        print(f"  {key}: {'OK' if value else 'CHECK'}")


def export_preview(path: str, out_dir: str, camera: Optional[str] = None, video: bool = False, fps: float = 10.0) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for preview export")
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(path, "r") as root:
        image_group = root.get("observations/images")
        if image_group is None:
            raise RuntimeError("episode has no observations/images group")
        camera_names = list(image_group.keys())
        selected = camera or camera_names[0]
        images = image_group[selected][:]

    indices = np.linspace(0, len(images) - 1, min(12, len(images)), dtype=int)
    for idx in indices:
        rgb = images[idx]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"{selected}_{idx:06d}.jpg"), bgr)

    if video:
        height, width = images.shape[1:3]
        out_path = os.path.join(out_dir, f"{selected}_preview.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {out_path}")
        for rgb in images:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="Path to episode_N.hdf5")
    parser.add_argument("--camera", default=None, help="Camera name to inspect/export; defaults to the first camera")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report")
    parser.add_argument("--export-dir", default=None, help="Directory for preview JPGs")
    parser.add_argument("--export-video", action="store_true", help="Also export an MP4 preview; requires --export-dir")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()

    result = analyze_episode(args.episode, args.camera)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)

    if args.export_dir:
        export_preview(args.episode, args.export_dir, args.camera, args.export_video, args.video_fps)
        print(f"preview exported to {args.export_dir}")


if __name__ == "__main__":
    main()
