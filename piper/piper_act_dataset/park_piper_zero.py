#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move one or more Piper arms back to the zero pose.

This is intended for dataset collection: after every trajectory, return the
slave arm to the same known starting pose so the next demonstration starts from
a consistent state.
"""

import argparse
from typing import Iterable

import numpy as np

from play_episode_piper import (
    connect_piper,
    enable_arm,
    ensure_can_mode,
    get_feedback_vector,
    move_to_start,
)


ZERO_TARGET = np.zeros(7, dtype=np.float32)


def park_can_to_zero(
    can_name: str,
    seconds: float,
    move_speed: int,
    include_gripper: bool,
    gripper_effort: int,
    timeout: float,
    try_can_mode: bool,
    dry_run: bool = False,
) -> None:
    print(f"Parking {can_name} to zero pose over {seconds:.1f}s")
    if dry_run:
        print(f"  dry-run target: {ZERO_TARGET}")
        return

    piper = connect_piper(can_name)
    ensure_can_mode(piper, move_speed, timeout, try_can_mode)
    enable_arm(piper, move_speed, include_gripper, timeout)
    current = get_feedback_vector(piper)
    print(f"  current: {np.round(current, 5)}")
    print(f"  target:  {np.round(ZERO_TARGET, 5)}")
    move_to_start(
        piper,
        current,
        ZERO_TARGET,
        seconds,
        move_speed,
        include_gripper,
        gripper_effort,
    )
    print(f"Parking complete: {can_name}")


def park_many_to_zero(
    can_names: Iterable[str],
    seconds: float,
    move_speed: int,
    include_gripper: bool,
    gripper_effort: int,
    timeout: float,
    try_can_mode: bool,
    dry_run: bool = False,
) -> None:
    for can_name in can_names:
        park_can_to_zero(
            can_name,
            seconds,
            move_speed,
            include_gripper,
            gripper_effort,
            timeout,
            try_can_mode,
            dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", action="append", required=True, help="Slave arm CAN interface. May be repeated.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Interpolation time to zero pose")
    parser.add_argument("--move-speed", type=int, default=20, help="Piper move speed percentage")
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--no-gripper", action="store_true", help="Move joints only")
    parser.add_argument("--try-can-mode", action="store_true", help="Try switching to CAN mode before parking")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        park_many_to_zero(
            args.can,
            args.seconds,
            args.move_speed,
            not args.no_gripper,
            args.gripper_effort,
            args.timeout,
            args.try_can_mode,
            args.dry_run,
        )
    except Exception as exc:
        print()
        print("Zero-pose parking failed.")
        print(f"  Error: {exc}")
        print("  Most common causes:")
        print("  - CAN adapter is up, but the arm is not powered or not connected to that CAN bus.")
        print("  - Wrong CAN interface was selected; try CAN=can1 or check wiring.")
        print("  - The arm/controller is not acknowledging outgoing CAN frames.")
        print("  - Another master-slave controller is holding the arm in a mode that rejects CAN control.")
        print("  Quick checks:")
        print("  - ip -statistics -details link show <can>")
        print("  - candump <can> should show Piper feedback frames while the arm is powered.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
