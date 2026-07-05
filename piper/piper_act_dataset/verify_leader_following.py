#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify that follower feedback changes while the operator moves the leader arm.

This is a supervised hardware check for the shared-can0 collection workflow.
It does not prove geometric direction; the operator still needs to watch that
the follower tracks the leader correctly. It does prove that the follower
feedback stream used by the HDF5 recorder is live and moving after restore.
"""

import argparse
import math
import time

import numpy as np


RAD_PER_MILLI_DEG = math.pi / 180000.0


def connect_piper(can_name: str):
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.1)
    return piper


def read_joint_feedback_rad(piper) -> tuple:
    joint_msg = piper.GetArmJointMsgs()
    state = joint_msg.joint_state
    joints = np.array([getattr(state, f"joint_{idx}") for idx in range(1, 7)], dtype=np.float32)
    return joints * RAD_PER_MILLI_DEG, float(joint_msg.Hz)


def verify_leader_following(
    can_name: str,
    duration: float,
    sample_rate: float,
    min_joint_range: float,
    dry_run: bool = False,
) -> None:
    print("Leader-follower motion verification:")
    print(f"  can={can_name}")
    print(f"  duration={duration}")
    print(f"  sample_rate={sample_rate}")
    print(f"  min_joint_range_rad={min_joint_range}")

    if dry_run:
        print("  dry-run: skipped feedback sampling")
        return

    input(
        "Press Enter, then gently move the leader arm for the verification window. "
        "Watch that the follower moves in the correct direction."
    )

    piper = connect_piper(can_name)
    dt = 1.0 / sample_rate
    deadline = time.time() + duration
    samples = []
    live_feedback_seen = False

    while time.time() < deadline:
        joints, hz = read_joint_feedback_rad(piper)
        samples.append(joints)
        live_feedback_seen = live_feedback_seen or hz > 0
        time.sleep(dt)

    if not samples:
        raise RuntimeError("no follower feedback samples were collected")
    if not live_feedback_seen:
        raise RuntimeError("joint feedback Hz stayed at 0; follower feedback may not be live")

    data = np.stack(samples)
    ranges = np.ptp(data, axis=0)
    max_range = float(np.max(ranges))
    print("  joint_ranges_rad: {}".format(" ".join(f"{value:.6f}" for value in ranges)))
    print(f"  max_joint_range_rad={max_range:.6f}")
    if max_range < min_joint_range:
        raise RuntimeError(
            "follower feedback did not move enough after restore: "
            f"max_joint_range_rad={max_range:.6f} < {min_joint_range:.6f}"
        )
    print("  OK: follower feedback changed while leader was moved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=float, default=25.0)
    parser.add_argument("--min-joint-range", type=float, default=0.03)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        verify_leader_following(
            args.can,
            args.duration,
            args.sample_rate,
            args.min_joint_range,
            args.dry_run,
        )
    except Exception as exc:
        print()
        print("Leader-follower motion verification failed.")
        print(f"  Error: {exc}")
        print("  Check that ReqMasterArmMoveToHome(0) restored teaching mode and that the leader is unlocked.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
