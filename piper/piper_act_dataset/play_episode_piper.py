#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safely replay a Piper ACT HDF5 episode on a real Piper arm.

The default replay source is /observations/qpos because that is the executed
follower trajectory. /action is available for debugging, but in this dataset it
is usually qpos shifted one frame forward for training labels.
"""

import argparse
import math
import time
from typing import Optional, Tuple

import h5py
import numpy as np
from piper_sdk import C_PiperInterface_V2


MILLI_DEG_PER_RAD = 180000.0 / math.pi


def load_trajectory(
    episode: str,
    source: str,
    side: str,
    max_frames: Optional[int],
    start_frame: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    with h5py.File(episode, "r") as root:
        data = root["observations/qpos"][:] if source == "qpos" else root["action"][:]
        timestamps = root["observations/timestamp_ns"][:] if "observations/timestamp_ns" in root else None
        fps = float(root.attrs.get("fps", 50.0))

    if side == "left":
        track = data[:, :7]
    elif side == "right":
        track = data[:, 7:14]
    else:
        raise ValueError(f"unsupported side: {side}")

    end = None if max_frames is None else start_frame + max_frames
    track = track[start_frame:end]
    if timestamps is not None:
        timestamps = timestamps[start_frame:end]
    if len(track) == 0:
        raise RuntimeError("selected trajectory has no frames")
    return track.astype(np.float32), timestamps, fps


def connect_piper(can_name: str) -> C_PiperInterface_V2:
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.1)
    return piper


def get_feedback_vector(piper: C_PiperInterface_V2) -> np.ndarray:
    joint_state = piper.GetArmJointMsgs().joint_state
    joints = np.array([getattr(joint_state, f"joint_{idx}") for idx in range(1, 7)], dtype=np.float32)
    joints = joints / MILLI_DEG_PER_RAD
    gripper = float(piper.GetArmGripperMsgs().gripper_state.grippers_angle) / 1e6
    return np.concatenate([joints, np.array([gripper], dtype=np.float32)]).astype(np.float32)


def send_target(
    piper: C_PiperInterface_V2,
    target: np.ndarray,
    move_speed: int,
    include_gripper: bool,
    gripper_effort: int,
) -> None:
    joints = [int(round(value * MILLI_DEG_PER_RAD)) for value in target[:6]]
    piper.MotionCtrl_2(0x01, 0x01, move_speed, 0x00)
    piper.JointCtrl(*joints)
    if include_gripper:
        piper.GripperCtrl(int(round(float(target[6]) * 1e6)), gripper_effort, 0x01, 0x00)


def ensure_can_mode(piper: C_PiperInterface_V2, move_speed: int, timeout: float, try_can_mode: bool) -> None:
    status = piper.GetArmStatus().arm_status
    if status.ctrl_mode == 1:
        return
    if not try_can_mode:
        raise RuntimeError(
            f"arm ctrl_mode is {status.ctrl_mode}, not CAN mode 1. "
            "Exit teaching/master-slave mode first, or retry with --try-can-mode."
        )

    print(f"Arm ctrl_mode is {status.ctrl_mode}; trying to exit teaching mode before CAN control.")
    piper.EmergencyStop(0x01)
    time.sleep(1.0)
    piper.EmergencyStop(0x02)
    time.sleep(1.0)

    deadline = time.time() + timeout
    while time.time() < deadline:
        piper.ModeCtrl(0x01, 0x01, move_speed, 0x00)
        time.sleep(0.05)
        if piper.GetArmStatus().arm_status.ctrl_mode == 1:
            return
    raise RuntimeError(
        "failed to switch arm to CAN mode. If the log shows SEND_MESSAGE_FAILED, "
        "the CAN socket could not transmit frames; check arm power, CAN wiring, "
        "the selected CAN interface, and whether candump shows Piper feedback."
    )


def enable_arm(piper: C_PiperInterface_V2, move_speed: int, include_gripper: bool, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if piper.EnablePiper():
            if include_gripper:
                piper.GripperCtrl(0, 1000, 0x01, 0x00)
            piper.ModeCtrl(0x01, 0x01, move_speed, 0x00)
            return
        time.sleep(0.05)
    raise RuntimeError("failed to enable Piper arm")


def move_to_start(
    piper: C_PiperInterface_V2,
    current: np.ndarray,
    first: np.ndarray,
    seconds: float,
    move_speed: int,
    include_gripper: bool,
    gripper_effort: int,
) -> None:
    steps = max(2, int(seconds * 50))
    for idx in range(1, steps + 1):
        alpha = idx / float(steps)
        target = current * (1.0 - alpha) + first * alpha
        send_target(piper, target, move_speed, include_gripper, gripper_effort)
        time.sleep(1.0 / 50.0)


def replay(
    piper: C_PiperInterface_V2,
    track: np.ndarray,
    timestamps_ns: Optional[np.ndarray],
    fps: float,
    speed_scale: float,
    move_speed: int,
    include_gripper: bool,
    gripper_effort: int,
) -> None:
    default_dt = 1.0 / fps
    for idx, target in enumerate(track):
        step_start = time.monotonic()
        send_target(piper, target, move_speed, include_gripper, gripper_effort)
        if idx < len(track) - 1:
            if timestamps_ns is not None and len(timestamps_ns) > idx + 1:
                dt = max(float(timestamps_ns[idx + 1] - timestamps_ns[idx]) / 1e9, 0.0)
            else:
                dt = default_dt
            sleep_time = max((dt / speed_scale) - (time.monotonic() - step_start), 0.0)
            time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="Path to episode_N.hdf5")
    parser.add_argument("--can", default="can0", help="SocketCAN interface")
    parser.add_argument("--source", choices=["qpos", "action"], default="qpos")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--speed-scale", type=float, default=0.5, help="0.5 means half-speed replay")
    parser.add_argument("--move-speed", type=int, default=20, help="Piper move speed percentage")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--no-gripper", action="store_true", help="Replay joints only")
    parser.add_argument("--start-threshold-rad", type=float, default=0.35)
    parser.add_argument("--move-to-start", action="store_true", help="Slowly interpolate to the first recorded pose")
    parser.add_argument("--move-to-start-seconds", type=float, default=5.0)
    parser.add_argument("--try-can-mode", action="store_true", help="Try switching to CAN mode if not already there")
    parser.add_argument("--allow-static", action="store_true", help="Allow replaying a trajectory with no detected motion")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--yes", action="store_true", help="Do not ask for final Enter confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Load and print replay plan without touching CAN")
    args = parser.parse_args()

    include_gripper = not args.no_gripper
    track, timestamps_ns, fps = load_trajectory(args.episode, args.source, args.side, args.max_frames, args.start_frame)
    duration = (len(track) - 1) / fps if timestamps_ns is None or len(timestamps_ns) < 2 else float(timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    print(
        f"Loaded {len(track)} frames from {args.episode}; source={args.source}, side={args.side}, "
        f"duration={duration:.2f}s, replay_speed_scale={args.speed_scale}"
    )
    print(f"First target: {np.round(track[0], 5)}")
    print(f"Last target:  {np.round(track[-1], 5)}")
    ranges = np.ptp(track, axis=0)
    print(f"Selected range: {np.round(ranges, 6)}")
    if not args.allow_static and not np.any(ranges[:6] > 1e-3) and ranges[6] <= 1e-4:
        raise SystemExit(
            "Selected trajectory has no detected arm/gripper motion. "
            "Refusing replay because this is likely an invalid recording. "
            "Use --allow-static only if you intentionally want to replay a static pose."
        )

    if args.dry_run:
        return

    piper = connect_piper(args.can)
    ensure_can_mode(piper, args.move_speed, args.timeout, args.try_can_mode)
    enable_arm(piper, args.move_speed, include_gripper, args.timeout)

    current = get_feedback_vector(piper)
    joint_distance = float(np.max(np.abs(current[:6] - track[0, :6])))
    gripper_distance = float(abs(current[6] - track[0, 6]))
    print(f"Current pose: {np.round(current, 5)}")
    print(f"Start distance: max_joint={joint_distance:.4f} rad, gripper={gripper_distance:.4f} m")
    if joint_distance > args.start_threshold_rad and not args.move_to_start:
        raise SystemExit(
            "Current arm pose is far from the first recorded pose. "
            "Move the arm closer manually, or retry with --move-to-start."
        )

    if args.move_to_start:
        if not args.yes:
            input("Press Enter to slowly move to the first recorded pose.")
        move_to_start(
            piper,
            current,
            track[0],
            args.move_to_start_seconds,
            args.move_speed,
            include_gripper,
            args.gripper_effort,
        )

    if not args.yes:
        input("Press Enter to replay the selected trajectory.")
    replay(piper, track, timestamps_ns, fps, args.speed_scale, args.move_speed, include_gripper, args.gripper_effort)
    print("Replay finished. Motors remain enabled; use prepare_next_collection.sh when ready to hand-guide the next demo.")


if __name__ == "__main__":
    main()
