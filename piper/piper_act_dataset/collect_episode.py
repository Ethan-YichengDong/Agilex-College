#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience wrapper for collecting Piper ACT episodes.

It performs light preflight checks, calls record_episodes_piper.py's capture
logic, runs preview_episode.py quality checks, and writes a JSON sidecar so each
episode can be marked keep/reject without renaming or deleting data.
"""

import argparse
import json
import os
import subprocess
from types import SimpleNamespace
from typing import List, Optional

from record_episodes_piper import CameraReader, CameraSpec, capture_episode, parse_camera_specs
from preview_episode import analyze_episode, export_preview, print_report


def check_can_interface(can_name: Optional[str]) -> None:
    if not can_name:
        return
    try:
        result = subprocess.run(
            ["ip", "-details", "link", "show", can_name],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to inspect {can_name}: {exc.stderr.strip() or exc}") from exc

    output = result.stdout
    if "state UP" not in output and "<" in output and "UP" not in output.split(">", 1)[0]:
        raise RuntimeError(f"{can_name} is not UP. Bring it up with: sudo ip link set {can_name} up type can bitrate 1000000")
    if "bitrate 1000000" not in output:
        raise RuntimeError(f"{can_name} does not report bitrate 1000000. Current ip output:\n{output}")
    print(f"CAN preflight passed: {can_name} is UP at 1000000 bps")


def check_cameras(camera_values: List[str], width: int, height: int) -> None:
    if not camera_values:
        return
    specs = parse_camera_specs(camera_values)
    reader = CameraReader(specs, width, height)
    try:
        images = reader.read()
        for name, image in images.items():
            print(
                f"Camera preflight {name}: shape={image.shape}, dtype={image.dtype}, "
                f"min={int(image.min())}, max={int(image.max())}, mean={float(image.mean()):.2f}"
            )
    finally:
        reader.close()


def build_capture_args(args: argparse.Namespace, dataset_dir: str) -> SimpleNamespace:
    episode_len = args.episode_len if args.episode_len is not None else int(round(args.duration * args.fps))
    return SimpleNamespace(
        dataset_dir=dataset_dir,
        episode_idx=args.episode_idx,
        episode_len=episode_len,
        fps=args.fps,
        overwrite=args.overwrite,
        pair_mode=args.pair_mode,
        action_source=args.action_source,
        left_slave_can=args.left_slave_can,
        right_slave_can=args.right_slave_can,
        left_master_can=args.left_master_can,
        right_master_can=args.right_master_can,
        camera=args.camera,
        image_width=args.image_width,
        image_height=args.image_height,
        print_every=args.print_every,
    )


def sidecar_path(episode_path: str) -> str:
    root, _ = os.path.splitext(episode_path)
    return root + ".qa.json"


def write_sidecar(episode_path: str, analysis: dict, keep: Optional[bool], note: str) -> None:
    payload = {
        "episode": episode_path,
        "keep": keep,
        "note": note,
        "analysis": analysis,
    }
    path = sidecar_path(episode_path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"QA sidecar written: {path}")


def ask_keep() -> tuple:
    answer = input("Keep this episode? [y/N]: ").strip().lower()
    keep = answer in ("y", "yes")
    note = input("Optional note: ").strip()
    return keep, note


def hard_quality_warning(analysis: dict) -> bool:
    flags = analysis.get("quality_flags", {})
    hard_failures = []
    for key in ("has_robot_rows", "some_arm_motion", "timing_near_50hz"):
        if key in flags and not flags[key]:
            hard_failures.append(key)
    if "image_nonblack" in flags and not flags["image_nonblack"]:
        hard_failures.append("image_nonblack")
    if not hard_failures:
        return False

    print()
    print("TECHNICAL WARNING: this episode should usually be rejected.")
    print("Failed quality flags: {}".format(", ".join(hard_failures)))
    print("If this was meant to be a real demonstration, fix the hardware/setup and record again.")
    print()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="datasets/piper_act_collection")
    parser.add_argument("--task", default=None, help="Optional task subdirectory under --dataset-dir")
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--episode-len", type=int, default=None, help="Override duration-derived frame count")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pair-mode", choices=["single", "dual"], default="single")
    parser.add_argument("--action-source", choices=["slave_next_qpos", "slave_current_qpos", "master_ctrl"], default="slave_next_qpos")
    parser.add_argument("--left-slave-can", default="can0")
    parser.add_argument("--right-slave-can", default=None)
    parser.add_argument("--left-master-can", default=None)
    parser.add_argument("--right-master-can", default=None)
    parser.add_argument("--camera", action="append", default=[], help="name=device, e.g. cam_high=/dev/video0")
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--export-preview", action="store_true")
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--no-ask-keep", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved config and run preflight only")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.dataset_dir, args.task) if args.task else args.dataset_dir
    capture_args = build_capture_args(args, dataset_dir)
    print("Resolved collection config:")
    print(f"  dataset_dir={capture_args.dataset_dir}")
    print(f"  episode_len={capture_args.episode_len}")
    print(f"  fps={capture_args.fps}")
    print(f"  can={capture_args.left_slave_can}")
    print(f"  cameras={capture_args.camera or 'none'}")
    print(f"  image_size={capture_args.image_width}x{capture_args.image_height}")

    if not args.skip_preflight:
        check_can_interface(args.left_slave_can)
        if args.pair_mode == "dual":
            check_can_interface(args.right_slave_can)
        check_cameras(args.camera, args.image_width, args.image_height)

    if args.dry_run:
        print("Dry run complete; no episode recorded.")
        return

    episode_path = capture_episode(capture_args)
    print(f"Recorded: {episode_path}")

    analysis = analyze_episode(episode_path)
    print_report(analysis)
    hard_failed = hard_quality_warning(analysis)

    if args.export_preview:
        preview_dir = args.preview_dir or os.path.join(os.path.dirname(episode_path), "previews", os.path.splitext(os.path.basename(episode_path))[0])
        export_preview(episode_path, preview_dir, video=args.export_video)

    if args.no_ask_keep:
        keep = False if hard_failed else None
        note = "auto technical reject" if hard_failed else ""
    else:
        keep, note = ask_keep()
    write_sidecar(episode_path, analysis, keep, note)


if __name__ == "__main__":
    main()
