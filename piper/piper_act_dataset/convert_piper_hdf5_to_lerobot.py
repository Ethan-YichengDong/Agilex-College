#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Piper ACT-style HDF5 episodes to an openpi/LeRobot dataset.

Input layout expected by default:

data/raw/<task_name>/successful/episode_N/episode_N.hdf5
data/raw/<task_name>/successful/episode_N/episode_N.qa.json

The HDF5 files are expected to follow the ACT/ALOHA convention used by this
repository:

observations/qpos, observations/qvel, observations/effort, action: [T, 14]
observations/images/<camera>: [T, H, W, 3], RGB uint8

Example:

python convert_piper_hdf5_to_lerobot.py \
  --raw-dir data/raw \
  --repo-id local/piper_bell \
  --lerobot-home data/lerobot \
  --fps 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


JOINT_NAMES = [
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

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Root containing raw task folders")
    parser.add_argument("--repo-id", default="local/piper_bell", help="LeRobot repo id / local dataset name")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path("data/lerobot"),
        help="Where LeRobot should write the dataset. The dataset is created under <lerobot-home>/<repo-id>.",
    )
    parser.add_argument("--fps", type=int, default=50, help="Dataset FPS used by LeRobot")
    parser.add_argument("--robot-type", default="agilex_piper", help="LeRobot robot_type metadata")
    parser.add_argument(
        "--camera",
        action="append",
        choices=DEFAULT_CAMERAS,
        default=None,
        help="Camera to include. May be repeated. Defaults to all three Piper cameras if present.",
    )
    parser.add_argument(
        "--image-dtype",
        choices=("image", "video"),
        default="image",
        help="LeRobot image feature dtype. openpi's examples use image; LeRobot may still encode videos internally.",
    )
    parser.add_argument("--no-videos", action="store_true", help="Disable LeRobot video writing")
    parser.add_argument("--include-qvel", action="store_true", help="Also store observations/qvel")
    parser.add_argument("--include-effort", action="store_true", help="Also store observations/effort")
    parser.add_argument("--include-rejected", action="store_true", help="Include episodes whose qa.json has keep=false")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Write a LeRobot v3-style parquet/meta dataset without importing the lerobot package.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for embedded lightweight image columns.",
    )
    parser.add_argument(
        "--include-image-stats",
        action="store_true",
        help="Compute image stats in metadata. Slower and more memory intensive; disabled by default.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only convert the first N valid episodes. Useful for lightweight smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing a LeRobot dataset")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output dataset first")
    return parser.parse_args()


def task_from_episode_path(path: Path, raw_dir: Path) -> str:
    rel = path.relative_to(raw_dir)
    task_name = rel.parts[0]
    return task_name.replace("_", " ")


def qa_path_for_hdf5(path: Path) -> Path:
    return path.with_suffix(".qa.json")


def load_qa(path: Path) -> dict[str, Any]:
    qa_path = qa_path_for_hdf5(path)
    if not qa_path.exists():
        return {}
    with qa_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_episode_files(raw_dir: Path, include_rejected: bool) -> tuple[list[Path], list[dict[str, Any]]]:
    candidates = sorted(raw_dir.glob("*/successful/episode_*/episode_*.hdf5"))
    selected: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for path in candidates:
        qa = load_qa(path)
        if qa and qa.get("keep") is False and not include_rejected:
            skipped.append({"path": str(path), "reason": "qa_keep_false"})
            continue
        selected.append(path)
    return selected, skipped


def require_dataset(root: h5py.File, key: str) -> h5py.Dataset:
    if key not in root:
        raise ValueError(f"missing dataset {key}")
    dataset = root[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{key} is not a dataset")
    return dataset


def validate_episode(path: Path, cameras: tuple[str, ...]) -> dict[str, Any]:
    with h5py.File(path, "r") as root:
        qpos = require_dataset(root, "observations/qpos")
        action = require_dataset(root, "action")

        if qpos.ndim != 2 or qpos.shape[1] != 14:
            raise ValueError(f"observations/qpos must be [T, 14], got {qpos.shape}")
        if action.shape != qpos.shape:
            raise ValueError(f"action shape {action.shape} does not match qpos {qpos.shape}")
        if qpos.shape[0] < 2:
            raise ValueError(f"episode is too short: {qpos.shape[0]} frames")

        info: dict[str, Any] = {
            "path": str(path),
            "frames": int(qpos.shape[0]),
            "attrs": {key: json_scalar(value) for key, value in root.attrs.items()},
            "cameras": {},
        }

        if "observations/timestamp_ns" in root:
            timestamps = root["observations/timestamp_ns"][:]
            if timestamps.shape != (qpos.shape[0],):
                raise ValueError(f"timestamp_ns shape mismatch: {timestamps.shape}")
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError("timestamp_ns is not strictly increasing")
            dt_ms = np.diff(timestamps.astype(np.float64)) / 1e6
            info["timing_ms"] = {
                "mean": float(dt_ms.mean()),
                "min": float(dt_ms.min()),
                "max": float(dt_ms.max()),
                "std": float(dt_ms.std()),
            }

        for camera in cameras:
            key = f"observations/images/{camera}"
            images = require_dataset(root, key)
            if images.ndim != 4:
                raise ValueError(f"{key} must be [T, H, W, C], got {images.shape}")
            if images.shape[0] != qpos.shape[0]:
                raise ValueError(f"{key} frame count {images.shape[0]} does not match qpos {qpos.shape[0]}")
            if images.shape[-1] != 3:
                raise ValueError(f"{key} must have RGB channel last, got {images.shape}")
            if images.dtype != np.uint8:
                raise ValueError(f"{key} must be uint8, got {images.dtype}")
            info["cameras"][camera] = {"shape": list(images.shape), "dtype": str(images.dtype)}

        for optional in ("observations/qvel", "observations/effort"):
            if optional in root and root[optional].shape != qpos.shape:
                raise ValueError(f"{optional} shape {root[optional].shape} does not match qpos {qpos.shape}")

    return info


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_features(example_info: dict[str, Any], cameras: tuple[str, ...], image_dtype: str, include_qvel: bool, include_effort: bool) -> dict[str, Any]:
    features: dict[str, Any] = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [JOINT_NAMES],
        },
    }

    if include_qvel:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": [JOINT_NAMES],
        }
    if include_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": [JOINT_NAMES],
        }

    for camera in cameras:
        shape = example_info["cameras"][camera]["shape"]
        _, height, width, channels = shape
        features[f"observation.images.{camera}"] = {
            "dtype": image_dtype,
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    return features


def user_features(features: dict[str, Any]) -> dict[str, Any]:
    default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {key: value for key, value in features.items() if key not in default_keys}


def create_lerobot_dataset(
    repo_id: str,
    lerobot_home: Path,
    fps: int,
    robot_type: str,
    features: dict[str, Any],
    use_videos: bool,
    overwrite: bool,
):
    lerobot_home = lerobot_home.resolve()
    output_dir = lerobot_home / repo_id
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_processes=5,
        image_writer_threads=10,
    )


def add_episode(dataset: Any, path: Path, raw_dir: Path, cameras: tuple[str, ...], include_qvel: bool, include_effort: bool) -> None:
    task = task_from_episode_path(path, raw_dir)
    with h5py.File(path, "r") as root:
        state = root["observations/qpos"][:].astype(np.float32)
        action = root["action"][:].astype(np.float32)
        qvel = root["observations/qvel"][:].astype(np.float32) if include_qvel and "observations/qvel" in root else None
        effort = (
            root["observations/effort"][:].astype(np.float32)
            if include_effort and "observations/effort" in root
            else None
        )
        images_per_camera = {
            camera: root[f"observations/images/{camera}"][:]
            for camera in cameras
        }

    for idx in range(state.shape[0]):
        frame = {
            "observation.state": state[idx],
            "action": action[idx],
            "task": task,
        }
        for camera, images in images_per_camera.items():
            frame[f"observation.images.{camera}"] = images[idx]
        if qvel is not None:
            frame["observation.velocity"] = qvel[idx]
        if effort is not None:
            frame["observation.effort"] = effort[idx]
        dataset.add_frame(frame)

    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def maybe_consolidate(dataset: Any) -> None:
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "finalize"):
        dataset.finalize()


def lightweight_dataset_dir(lerobot_home: Path, repo_id: str) -> Path:
    return lerobot_home / repo_id


def data_file_path(output_dir: Path, episode_index: int) -> Path:
    return output_dir / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"


def episodes_file_path(output_dir: Path) -> Path:
    return output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def encode_jpeg_rgb(frame_rgb: np.ndarray, quality: int) -> bytes:
    import cv2

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to JPEG-encode image frame")
    return encoded.tobytes()


def fixed_size_list_array(values: np.ndarray, width: int):
    import pyarrow as pa

    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def scalar_array(value: int | float, length: int, dtype: str):
    import pyarrow as pa

    if dtype == "int64":
        return pa.array([int(value)] * length, type=pa.int64())
    if dtype == "float32":
        return pa.array([float(value)] * length, type=pa.float32())
    raise ValueError(f"unsupported scalar dtype {dtype}")


def image_array(images: np.ndarray, jpeg_quality: int):
    import pyarrow as pa

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    return pa.array(
        [{"bytes": encode_jpeg_rgb(frame, jpeg_quality), "path": None} for frame in images],
        type=image_type,
    )


def write_lightweight_episode(
    output_dir: Path,
    episode_index: int,
    dataset_from_index: int,
    task_index: int,
    path: Path,
    cameras: tuple[str, ...],
    include_qvel: bool,
    include_effort: bool,
    jpeg_quality: int,
    fps: int,
    include_image_stats: bool,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with h5py.File(path, "r") as root:
        state = root["observations/qpos"][:].astype(np.float32)
        action = root["action"][:].astype(np.float32)
        length = int(state.shape[0])
        stats = {
            "observation.state": numeric_stats(state),
            "action": numeric_stats(action),
        }

        columns: dict[str, Any] = {
            "observation.state": fixed_size_list_array(state, 14),
            "action": fixed_size_list_array(action, 14),
            "timestamp": pa.array(np.arange(length, dtype=np.float32) / np.float32(fps), type=pa.float32()),
            "frame_index": pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
            "episode_index": scalar_array(episode_index, length, "int64"),
            "index": pa.array(np.arange(dataset_from_index, dataset_from_index + length, dtype=np.int64), type=pa.int64()),
            "task_index": scalar_array(task_index, length, "int64"),
        }

        if include_qvel and "observations/qvel" in root:
            qvel = root["observations/qvel"][:].astype(np.float32)
            columns["observation.velocity"] = fixed_size_list_array(qvel, 14)
            stats["observation.velocity"] = numeric_stats(qvel)
        if include_effort and "observations/effort" in root:
            effort = root["observations/effort"][:].astype(np.float32)
            columns["observation.effort"] = fixed_size_list_array(effort, 14)
            stats["observation.effort"] = numeric_stats(effort)

        for camera in cameras:
            images = root[f"observations/images/{camera}"][:]
            columns[f"observation.images.{camera}"] = image_array(images, jpeg_quality)
            if include_image_stats:
                stats[f"observation.images.{camera}"] = image_stats(images[sample_indices(images.shape[0])])

    table = pa.table(columns)
    target = data_file_path(output_dir, episode_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="snappy", use_dictionary=True)

    return {
        "episode_index": episode_index,
        "tasks": [],
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": episode_index,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_from_index + length,
        "_stats": stats,
    }


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def sample_indices(data_len: int) -> np.ndarray:
    min_num_samples = min(100, data_len)
    num_samples = max(min_num_samples, min(int(data_len**0.75), 10_000))
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int)


def auto_downsample_chw(images: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, _, height, width = images.shape
    if max(width, height) < max_size_threshold:
        return images
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return images[:, :, ::downsample_factor, ::downsample_factor]


def image_stats(images_hwc: np.ndarray) -> dict[str, Any]:
    images_chw = images_hwc.transpose(0, 3, 1, 2)
    images_chw = auto_downsample_chw(images_chw)
    channels = images_chw.transpose(0, 2, 3, 1).reshape(-1, images_chw.shape[1]).astype(np.float32) / 255.0
    if channels.shape[0] > 200_000:
        quantile_channels = channels[np.linspace(0, channels.shape[0] - 1, 200_000).astype(np.int64)]
    else:
        quantile_channels = channels

    def channel_value(values: np.ndarray) -> list[list[list[float]]]:
        return values.astype(float).reshape(-1, 1, 1).tolist()

    return {
        "min": channel_value(channels.min(axis=0)),
        "max": channel_value(channels.max(axis=0)),
        "mean": channel_value(channels.mean(axis=0)),
        "std": channel_value(channels.std(axis=0)),
        "count": [int(images_chw.shape[0])],
        "q01": channel_value(np.quantile(quantile_channels, 0.01, axis=0)),
        "q10": channel_value(np.quantile(quantile_channels, 0.10, axis=0)),
        "q50": channel_value(np.quantile(quantile_channels, 0.50, axis=0)),
        "q90": channel_value(np.quantile(quantile_channels, 0.90, axis=0)),
        "q99": channel_value(np.quantile(quantile_channels, 0.99, axis=0)),
    }


def aggregate_feature_stats(feature_stats: list[dict[str, Any]]) -> dict[str, Any]:
    means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats in feature_stats])
    variances = np.stack([np.asarray(stats["std"], dtype=np.float64) ** 2 for stats in feature_stats])
    counts = np.stack([np.asarray(stats["count"], dtype=np.float64) for stats in feature_stats])
    total_count = counts.sum(axis=0)
    expanded_counts = counts
    while expanded_counts.ndim < means.ndim:
        expanded_counts = np.expand_dims(expanded_counts, axis=-1)

    total_mean = (means * expanded_counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * expanded_counts).sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([np.asarray(stats["min"], dtype=np.float64) for stats in feature_stats]), axis=0),
        "max": np.max(np.stack([np.asarray(stats["max"], dtype=np.float64) for stats in feature_stats]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    for q in DEFAULT_QUANTILES:
        key = f"q{int(q * 100):02d}"
        quantile_values = np.stack([np.asarray(stats[key], dtype=np.float64) for stats in feature_stats])
        aggregated[key] = (quantile_values * expanded_counts).sum(axis=0) / total_count

    return {key: value.astype(float).tolist() for key, value in aggregated.items()}


def aggregate_episode_stats(episode_stats: dict[int, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    feature_keys = sorted({key for stats in episode_stats.values() for key in stats})
    return {
        key: aggregate_feature_stats([stats[key] for stats in episode_stats.values() if key in stats])
        for key in feature_keys
    }


def flatten_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for feature, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            flat[f"stats/{feature}/{stat_name}"] = value
    return flat


def episode_feature_stats(path: Path, cameras: tuple[str, ...], include_qvel: bool, include_effort: bool) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as root:
        stats["observation.state"] = numeric_stats(root["observations/qpos"][:].astype(np.float32))
        stats["action"] = numeric_stats(root["action"][:].astype(np.float32))
        if include_qvel and "observations/qvel" in root:
            stats["observation.velocity"] = numeric_stats(root["observations/qvel"][:].astype(np.float32))
        if include_effort and "observations/effort" in root:
            stats["observation.effort"] = numeric_stats(root["observations/effort"][:].astype(np.float32))
        for camera in cameras:
            images = root[f"observations/images/{camera}"]
            stats[f"observation.images.{camera}"] = image_stats(images[sample_indices(images.shape[0])])
    return stats


def write_lightweight_stats(
    output_dir: Path,
    valid_paths: list[Path],
    cameras: tuple[str, ...],
    include_qvel: bool,
    include_effort: bool,
) -> dict[int, dict[str, dict[str, Any]]]:
    episode_stats: dict[int, dict[str, dict[str, Any]]] = {}
    for episode_index, path in enumerate(tqdm(valid_paths, desc="Computing lightweight stats")):
        episode_stats[episode_index] = episode_feature_stats(path, cameras, include_qvel, include_effort)

    stats = aggregate_episode_stats(episode_stats)
    write_report(output_dir / "meta" / "stats.json", stats)
    return episode_stats


def write_lightweight_lerobot_dataset(
    repo_id: str,
    lerobot_home: Path,
    raw_dir: Path,
    valid_paths: list[Path],
    validated: list[dict[str, Any]],
    cameras: tuple[str, ...],
    fps: int,
    robot_type: str,
    image_dtype: str,
    include_qvel: bool,
    include_effort: bool,
    jpeg_quality: int,
    include_image_stats: bool,
    overwrite: bool,
) -> Path:
    import pandas as pd

    output_dir = lightweight_dataset_dir(lerobot_home, repo_id)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    features = make_features(validated[0], cameras, image_dtype, include_qvel, include_effort)
    tasks = sorted({task_from_episode_path(path, raw_dir) for path in valid_paths})
    task_to_index = {task: idx for idx, task in enumerate(tasks)}

    episodes = []
    episode_stats = {}
    dataset_index = 0
    for episode_index, path in enumerate(tqdm(valid_paths, desc="Writing lightweight LeRobot episodes")):
        task = task_from_episode_path(path, raw_dir)
        episode = write_lightweight_episode(
            output_dir=output_dir,
            episode_index=episode_index,
            dataset_from_index=dataset_index,
            task_index=task_to_index[task],
            path=path,
            cameras=cameras,
            include_qvel=include_qvel,
            include_effort=include_effort,
            jpeg_quality=jpeg_quality,
            fps=fps,
            include_image_stats=include_image_stats,
        )
        episode["tasks"] = [task]
        episode_stats[episode_index] = episode.pop("_stats")
        episodes.append(episode)
        dataset_index += int(episode["length"])

    meta_dir = output_dir / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks_df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task"))
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    for episode in episodes:
        episode.update(flatten_stats(episode_stats[int(episode["episode_index"])]))

    episodes_df = pd.DataFrame(episodes)
    episodes_df.to_parquet(episodes_file_path(output_dir), index=False)

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(valid_paths),
        "total_frames": dataset_index,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{len(valid_paths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }
    write_report(meta_dir / "info.json", info)
    write_report(meta_dir / "stats.json", aggregate_episode_stats(episode_stats))
    return output_dir


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    cameras = tuple(args.camera or DEFAULT_CAMERAS)

    hdf5_files, skipped = find_episode_files(raw_dir, args.include_rejected)
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 episodes found under {raw_dir}")

    validated = []
    rejected = list(skipped)
    for path in tqdm(hdf5_files, desc="Validating episodes"):
        try:
            validated.append(validate_episode(path, cameras))
        except Exception as exc:  # noqa: BLE001 - report all bad episodes and continue.
            rejected.append({"path": str(path), "reason": str(exc)})

    valid_paths = [Path(item["path"]) for item in validated]
    if not valid_paths:
        raise RuntimeError("All episodes failed validation")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        valid_paths = valid_paths[: args.max_episodes]
        validated = validated[: args.max_episodes]

    report = {
        "repo_id": args.repo_id,
        "raw_dir": str(raw_dir),
        "lerobot_home": str(args.lerobot_home.resolve()),
        "fps": args.fps,
        "robot_type": args.robot_type,
        "cameras": list(cameras),
        "num_valid_episodes": len(valid_paths),
        "num_rejected_episodes": len(rejected),
        "num_frames": int(sum(item["frames"] for item in validated)),
        "tasks": sorted({task_from_episode_path(path, raw_dir) for path in valid_paths}),
        "rejected": rejected,
        "episodes": validated,
    }

    report_path = args.lerobot_home / args.repo_id / "piper_conversion_report.json"
    if args.dry_run:
        write_report(Path("data/lerobot_dry_run_report.json"), report)
        print(f"Dry run OK: {len(valid_paths)} episodes, {report['num_frames']} frames")
        print("Report: data/lerobot_dry_run_report.json")
        return

    if args.lightweight:
        output_dir = write_lightweight_lerobot_dataset(
            repo_id=args.repo_id,
            lerobot_home=args.lerobot_home,
            raw_dir=raw_dir,
            valid_paths=valid_paths,
            validated=validated,
            cameras=cameras,
            fps=args.fps,
            robot_type=args.robot_type,
            image_dtype=args.image_dtype,
            include_qvel=args.include_qvel,
            include_effort=args.include_effort,
            jpeg_quality=args.jpeg_quality,
            include_image_stats=args.include_image_stats,
            overwrite=args.overwrite,
        )
        write_report(report_path, report)
        print(f"Converted {len(valid_paths)} episodes / {report['num_frames']} frames")
        print(f"Lightweight dataset: {output_dir}")
        print(f"Report: {report_path}")
        return

    features = user_features(make_features(validated[0], cameras, args.image_dtype, args.include_qvel, args.include_effort))
    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        lerobot_home=args.lerobot_home,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=not args.no_videos,
        overwrite=args.overwrite,
    )

    for path in tqdm(valid_paths, desc="Writing LeRobot episodes"):
        add_episode(dataset, path, raw_dir, cameras, args.include_qvel, args.include_effort)

    maybe_consolidate(dataset)
    write_report(report_path, report)
    print(f"Converted {len(valid_paths)} episodes / {report['num_frames']} frames")
    print(f"Dataset: {args.lerobot_home / args.repo_id}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
