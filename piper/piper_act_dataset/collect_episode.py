#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience wrapper for collecting Piper ACT episodes.

It performs light preflight checks, calls record_episodes_piper.py's capture
logic, runs technical quality checks, and writes a JSON sidecar so each
episode can be marked keep/reject without renaming or deleting data.
"""

import argparse
import json
import os
import subprocess
from types import SimpleNamespace
from typing import List, Optional

from park_piper_zero import park_many_to_zero
from record_episodes_piper import CameraReader, CameraSpec, capture_episode, parse_camera_specs
from request_master_home import check_master_home_support, request_master_home_cycle
from restore_leader_follower import restore_pairs
from preview_episode import analyze_episode, print_report


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DEFAULT_RAW_DIR = os.path.join(DATA_DIR, "raw")


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


def check_cameras(camera_values: List[str], width: int, height: int, camera_fps: float) -> None:
    if not camera_values:
        return
    specs = parse_camera_specs(camera_values)
    reader = CameraReader(specs, width, height, camera_fps)
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
        camera_fps=args.camera_fps,
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


def print_quality_feedback(analysis: dict) -> None:
    flags = analysis.get("quality_flags", {})
    checks = [key for key, value in flags.items() if not value]
    print()
    print("Quality feedback:")
    if checks:
        print("  CHECK flags: {}".format(", ".join(checks)))
    else:
        print("  CHECK flags: none")
    if all(key in analysis for key in ("left_min", "left_max", "left_range")):
        print("  left data min/max/range:")
        for name, v_min, v_max, v_range in zip(
            ("j1", "j2", "j3", "j4", "j5", "j6", "gripper"),
            analysis["left_min"],
            analysis["left_max"],
            analysis["left_range"],
        ):
            print(f"    {name}: min={v_min:.6f}, max={v_max:.6f}, range={v_range:.6f}")
    print("  Decision is manual: keep/reject is not forced by quality checks.")
    print()


def print_config(capture_args: SimpleNamespace, args: argparse.Namespace) -> None:
    print("Resolved collection config:")
    print(f"  dataset_dir={capture_args.dataset_dir}")
    print(f"  num_episodes={args.num_episodes}")
    print(f"  episode_idx={capture_args.episode_idx if capture_args.episode_idx is not None else 'next available'}")
    print(f"  episode_len={capture_args.episode_len}")
    print(f"  fps={capture_args.fps}")
    print(f"  can={capture_args.left_slave_can}")
    print(f"  cameras={capture_args.camera or 'none'}")
    print(f"  image_size={capture_args.image_width}x{capture_args.image_height}")
    print(f"  camera_fps={capture_args.camera_fps}")
    print(f"  prepare_before={args.prepare_before}")
    print(f"  park_after={args.park_after}")
    if args.prepare_before or args.park_after:
        print(f"  park_method={args.park_method}")
        if args.park_method == "master_home":
            print(f"  master_home_can={args.master_home_can}")
            print(f"  master_home_wait={args.master_home_wait}")
            print(f"  master_home_restore={args.master_home_restore}")
            print(f"  master_home_preflight={args.master_home_preflight}")
            print(f"  master_home_allow_unknown_firmware={args.master_home_allow_unknown_firmware}")
            print(f"  master_home_verify_zero={args.master_home_verify_zero}")
            if args.master_home_verify_zero:
                print(f"  master_home_zero_timeout={args.master_home_zero_timeout}")
                print(f"  master_home_joint_tolerance={args.master_home_joint_tolerance}")
                print(
                    "  master_home_gripper_tolerance={}".format(
                        args.master_home_gripper_tolerance
                        if args.master_home_gripper_tolerance is not None
                        else "disabled"
                    )
                )
        else:
            print(f"  park_seconds={args.park_seconds}")
            print(f"  park_try_can_mode={args.park_try_can_mode}")


def reset_to_start_pose(args: argparse.Namespace, reason: str) -> None:
    if args.park_method == "master_home":
        if args.pair_mode != "single":
            raise RuntimeError("PARK_METHOD=master_home is currently implemented for PAIR_MODE=single only")
        if not args.master_home_can:
            raise RuntimeError("PARK_METHOD=master_home requires --master-home-can")

        print()
        print(f"Requesting leader/follower pair zero return {reason}.")
        request_master_home_cycle(
            args.master_home_can,
            args.master_home_wait,
            args.master_home_restore,
            verify_zero=args.master_home_verify_zero,
            zero_timeout=args.master_home_zero_timeout,
            joint_tolerance=args.master_home_joint_tolerance,
            gripper_tolerance=args.master_home_gripper_tolerance,
        )
        return

    if args.park_method != "can_park":
        raise RuntimeError(f"unsupported park method: {args.park_method}")

    can_names = [args.left_slave_can]
    if args.pair_mode == "dual":
        can_names.append(args.right_slave_can)
    can_names = [name for name in can_names if name]
    if not can_names:
        raise RuntimeError("parking requested but no slave CAN interface was configured")

    print()
    print(f"Returning slave arm(s) to zero pose {reason}.")
    park_many_to_zero(
        can_names,
        args.park_seconds,
        args.park_move_speed,
        not args.park_no_gripper,
        args.park_gripper_effort,
        args.park_timeout,
        args.park_try_can_mode,
    )


def restore_leader_follower_after_park(args: argparse.Namespace) -> None:
    pairs = []
    if args.restore_role in ("leader", "both") and args.left_leader_can:
        pairs.append(("leader", args.left_leader_can))
    if args.restore_role in ("follower", "both") and args.left_follower_can:
        pairs.append(("follower", args.left_follower_can))
    if args.pair_mode == "dual":
        if args.restore_role in ("leader", "both") and args.right_leader_can:
            pairs.append(("leader", args.right_leader_can))
        if args.restore_role in ("follower", "both") and args.right_follower_can:
            pairs.append(("follower", args.right_follower_can))

    if not pairs:
        print("Leader-follower restore skipped: no leader/follower CAN ports configured.")
        return

    print()
    print("Restoring leader-follower teaching mode after parking.")
    restore_pairs(pairs, args.restore_dry_run)


def validate_restore_config(args: argparse.Namespace) -> None:
    if not args.restore_leader_follower:
        return

    checks = [
        ("left/single", args.left_leader_can, args.left_follower_can),
        ("right", args.right_leader_can, args.right_follower_can),
    ]
    for label, leader_can, follower_can in checks:
        if leader_can and follower_can and leader_can == follower_can:
            raise RuntimeError(
                "automatic leader-follower restore is unsafe for the {} pair because "
                "leader and follower are both configured on {}. A shared CAN role "
                "command is broadcast to both arms and can cause drop or reversed "
                "roles. Configure one physical arm at a time, or use separate CAN "
                "interfaces / validated CAN-ID offsets.".format(label, leader_can)
            )


def reset_before_collection(args: argparse.Namespace) -> None:
    try:
        reset_to_start_pose(args, "before the first trajectory")
        if args.restore_leader_follower:
            restore_leader_follower_after_park(args)
    except Exception as exc:
        print()
        print("PRE-COLLECTION RESET FAILED.")
        print(f"  Reset error: {exc}")
        print("  Do not start collection until both arms are returned to the common start pose.")
        raise


def collect_one_episode(args: argparse.Namespace, dataset_dir: str, offset: int) -> str:
    capture_args = build_capture_args(args, dataset_dir)
    if args.episode_idx is not None:
        capture_args.episode_idx = args.episode_idx + offset

    if args.num_episodes > 1:
        print()
        print(f"=== Episode {offset + 1}/{args.num_episodes} ===")

    episode_path = capture_episode(capture_args)
    print(f"Recorded: {episode_path}")

    analysis = analyze_episode(episode_path)
    print_report(analysis)
    print_quality_feedback(analysis)

    if args.no_ask_keep:
        keep = None
        note = ""
    else:
        keep, note = ask_keep()
    write_sidecar(episode_path, analysis, keep, note)

    if args.park_after:
        try:
            reset_to_start_pose(args, "before the next trajectory")
            if args.restore_leader_follower:
                restore_leader_follower_after_park(args)
        except Exception as exc:
            print()
            print("POST-EPISODE RESET FAILED.")
            print(f"  Episode was already saved: {episode_path}")
            print(f"  QA sidecar was already written: {sidecar_path(episode_path)}")
            print(f"  Reset error: {exc}")
            print("  Do not continue repeated collection until the arm is manually returned to the start pose.")
            raise
    return episode_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--task", default=None, help="Task subdirectory under --dataset-dir, e.g. press_ring")
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to collect in this run")
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
    parser.add_argument("--camera-fps", type=float, default=60.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-ask-keep", action="store_true")
    parser.add_argument("--prepare-before", action="store_true", help="Return arm pair to the start pose before episode 1")
    parser.add_argument("--park-after", action="store_true", help="Return slave arm(s) to zero after each episode")
    parser.add_argument("--park-method", choices=["master_home", "can_park"], default="can_park")
    parser.add_argument("--master-home-can", default=None, help="CAN interface for ReqMasterArmMoveToHome")
    parser.add_argument("--master-home-wait", type=float, default=6.0, help="Wait after ReqMasterArmMoveToHome(2)")
    parser.add_argument("--master-home-preflight", action="store_true", help="Check SDK/firmware support before collection")
    parser.add_argument("--master-home-firmware-timeout", type=float, default=3.0)
    parser.add_argument("--master-home-allow-unknown-firmware", action="store_true", help="Allow collection when firmware cannot be parsed")
    parser.add_argument("--no-master-home-restore", action="store_true", help="Do not send ReqMasterArmMoveToHome(0) after zero return")
    parser.add_argument("--master-home-verify-zero", action="store_true", help="Require follower feedback to be near zero after master_home reset")
    parser.add_argument("--master-home-zero-timeout", type=float, default=8.0)
    parser.add_argument("--master-home-joint-tolerance", type=float, default=0.08, help="Max absolute joint error in radians")
    parser.add_argument("--master-home-gripper-tolerance", type=float, default=None, help="Optional max absolute gripper error in meters")
    parser.add_argument("--park-seconds", type=float, default=5.0, help="Seconds used to interpolate back to zero")
    parser.add_argument("--park-move-speed", type=int, default=20, help="Piper move speed percentage for parking")
    parser.add_argument("--park-gripper-effort", type=int, default=1000)
    parser.add_argument("--park-no-gripper", action="store_true", help="Park joints only")
    parser.add_argument("--park-try-can-mode", action="store_true", help="Try switching slave arm(s) to CAN mode before parking")
    parser.add_argument("--park-timeout", type=float, default=5.0)
    parser.add_argument("--restore-leader-follower", action="store_true", help="Restore leader-follower mode after parking")
    parser.add_argument("--restore-role", choices=["leader", "follower", "both"], default="leader")
    parser.add_argument("--left-leader-can", default=None, help="CAN port for left/single leader input arm")
    parser.add_argument("--right-leader-can", default=None, help="CAN port for right leader input arm")
    parser.add_argument("--left-follower-can", default=None, help="CAN port for left/single follower output arm")
    parser.add_argument("--right-follower-can", default=None, help="CAN port for right follower output arm")
    parser.add_argument("--restore-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved config and run preflight only")
    args = parser.parse_args()
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be >= 1")
    if args.left_follower_can is None:
        args.left_follower_can = args.left_slave_can
    if args.right_follower_can is None:
        args.right_follower_can = args.right_slave_can
    if args.master_home_can is None:
        args.master_home_can = args.left_slave_can
    args.master_home_restore = not args.no_master_home_restore
    validate_restore_config(args)

    dataset_dir = os.path.join(args.dataset_dir, args.task) if args.task else args.dataset_dir
    capture_args = build_capture_args(args, dataset_dir)
    print_config(capture_args, args)

    if not args.skip_preflight:
        check_can_interface(args.left_slave_can)
        if args.pair_mode == "dual":
            check_can_interface(args.right_slave_can)
        if (args.prepare_before or args.park_after) and args.park_method == "master_home":
            check_can_interface(args.master_home_can)
            if args.master_home_preflight:
                check_master_home_support(
                    args.master_home_can,
                    args.master_home_firmware_timeout,
                    allow_unknown_firmware=args.master_home_allow_unknown_firmware,
                )
        check_cameras(args.camera, args.image_width, args.image_height, args.camera_fps)

    if args.dry_run:
        print("Dry run complete; no episode recorded.")
        return

    if args.prepare_before:
        reset_before_collection(args)

    for offset in range(args.num_episodes):
        collect_one_episode(args, dataset_dir, offset)


if __name__ == "__main__":
    main()
