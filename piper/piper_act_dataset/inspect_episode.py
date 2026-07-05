#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import h5py
import numpy as np


def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/")


def _require_dataset(root: h5py.File, name: str) -> h5py.Dataset:
    if name not in root:
        raise RuntimeError(f"missing dataset: {name}")
    obj = root[name]
    if not isinstance(obj, h5py.Dataset):
        raise RuntimeError(f"path is not a dataset: {name}")
    return obj


def _check_shape(name: str, array: np.ndarray, shape_tail: tuple) -> None:
    if array.ndim != 1 + len(shape_tail):
        raise RuntimeError(f"{name} rank mismatch: got shape {array.shape}")
    if tuple(array.shape[1:]) != shape_tail:
        raise RuntimeError(f"{name} shape mismatch: got {array.shape}, expected (*, {shape_tail})")


def validate_act_episode(
    root: h5py.File,
    expected_len: int,
    expected_cameras: list,
    require_images: bool,
    action_shift_tolerance: float,
) -> None:
    qpos_ds = _require_dataset(root, "observations/qpos")
    qvel_ds = _require_dataset(root, "observations/qvel")
    effort_ds = _require_dataset(root, "observations/effort")
    timestamp_ds = _require_dataset(root, "observations/timestamp_ns")
    action_ds = _require_dataset(root, "action")

    qpos = qpos_ds[:]
    qvel = qvel_ds[:]
    action = action_ds[:]
    effort = effort_ds[:]
    timestamps = timestamp_ds[:]

    _check_shape("observations/qpos", qpos, (14,))
    _check_shape("observations/qvel", qvel, (14,))
    _check_shape("observations/effort", effort, (14,))
    _check_shape("action", action, (14,))
    if timestamps.ndim != 1:
        raise RuntimeError(f"observations/timestamp_ns must be 1-D, got {timestamps.shape}")

    length = qpos.shape[0]
    if expected_len and length != expected_len:
        raise RuntimeError(f"episode length mismatch: got {length}, expected {expected_len}")
    if length < 2:
        raise RuntimeError(f"episode is too short: {length} rows")

    for name, array in (
        ("observations/qpos", qpos),
        ("observations/qvel", qvel),
        ("observations/effort", effort),
        ("action", action),
    ):
        if array.shape[0] != length:
            raise RuntimeError(f"{name} length mismatch: got {array.shape[0]}, expected {length}")
        if not np.issubdtype(array.dtype, np.floating):
            raise RuntimeError(f"{name} dtype must be floating, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains NaN or inf")

    if timestamps.shape[0] != length:
        raise RuntimeError(f"observations/timestamp_ns length mismatch: got {timestamps.shape[0]}, expected {length}")
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise RuntimeError(f"observations/timestamp_ns dtype must be integer, got {timestamps.dtype}")
    if np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("observations/timestamp_ns must be strictly increasing")

    state_dim = int(root.attrs.get("state_dim", 0))
    if state_dim != 14:
        raise RuntimeError(f"state_dim attr mismatch: got {state_dim}, expected 14")

    action_source = root.attrs.get("action_source", "")
    if isinstance(action_source, bytes):
        action_source = action_source.decode("utf-8", errors="replace")
    if action_source == "slave_next_qpos":
        max_err = float(np.max(np.abs(action[:-1] - qpos[1:])))
        if max_err > action_shift_tolerance:
            raise RuntimeError(
                f"action is not next qpos: max error {max_err:.9f} > {action_shift_tolerance}"
            )

    image_group = root.get("observations/images")
    if require_images and image_group is None:
        raise RuntimeError("missing image group: observations/images")
    if expected_cameras:
        if image_group is None:
            raise RuntimeError("expected cameras but observations/images is missing")
        missing = [name for name in expected_cameras if name not in image_group]
        if missing:
            raise RuntimeError(f"missing expected camera dataset(s): {missing}")

    if image_group is not None:
        for name, dataset in image_group.items():
            images = dataset[:]
            if images.ndim != 4 or images.shape[0] != length or images.shape[-1] != 3:
                raise RuntimeError(f"camera {name} shape must be [T,H,W,3], got {images.shape}")
            if images.dtype != np.uint8:
                raise RuntimeError(f"camera {name} dtype must be uint8, got {images.dtype}")

    print("ACT validation: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", help="Path to episode_N.hdf5")
    parser.add_argument("--validate-act", action="store_true", help="Fail if the episode is not a valid Piper ACT HDF5")
    parser.add_argument("--expected-len", type=int, default=0, help="Expected number of timesteps; 0 disables")
    parser.add_argument("--expected-camera", action="append", default=[], help="Camera name expected under observations/images; may be repeated")
    parser.add_argument("--require-images", action="store_true", help="Require observations/images to exist")
    parser.add_argument("--action-shift-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    with h5py.File(args.episode, "r") as root:
        print("attrs:")
        for key, value in root.attrs.items():
            print(f"  {key}: {value}")
        print("datasets:")
        root.visititems(visit)
        if args.validate_act:
            validate_act_episode(
                root,
                args.expected_len,
                args.expected_camera,
                args.require_images,
                args.action_shift_tolerance,
            )


if __name__ == "__main__":
    main()
