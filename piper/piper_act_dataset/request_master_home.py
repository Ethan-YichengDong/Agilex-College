#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Request Piper master-slave return-to-home actions.

This uses the SDK's ReqMasterArmMoveToHome command, available on firmware
V1.7-4 and later:

  mode 0: restore master-slave mode
  mode 1: leader/master arm return-to-zero
  mode 2: leader/master and follower/slave arms return-to-zero together
"""

import argparse
import math
import re
import time
from typing import List, Optional, Tuple


MODES = {
    "restore": 0,
    "leader_zero": 1,
    "both_zero": 2,
}

MIN_MASTER_HOME_FIRMWARE = (1, 7, 4)
RAD_PER_MILLI_DEG = math.pi / 180000.0


def connect_piper(can_name: str):
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.1)
    return piper


def parse_firmware_version(version: object) -> Optional[Tuple[int, int, int]]:
    if not isinstance(version, str):
        return None
    match = re.search(r"S-V(\d+)\.(\d+)-(\d+)", version)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def is_master_home_firmware(version: object) -> Optional[bool]:
    parsed = parse_firmware_version(version)
    if parsed is None:
        return None
    return parsed >= MIN_MASTER_HOME_FIRMWARE


def read_firmware_version(piper, timeout: float) -> object:
    piper.SearchPiperFirmwareVersion()
    deadline = time.time() + timeout
    version = piper.GetPiperFirmwareVersion()
    seen = []
    while time.time() < deadline:
        if version not in seen:
            seen.append(version)
        if parse_firmware_version(version) is not None:
            return version
        time.sleep(0.1)
        version = piper.GetPiperFirmwareVersion()
    if version not in seen:
        seen.append(version)
    if len(seen) > 1:
        print(f"  firmware responses observed: {seen}")
    return version


def check_master_home_support(
    can_name: str,
    timeout: float,
    dry_run: bool = False,
    allow_unknown_firmware: bool = False,
) -> None:
    from piper_sdk import C_PiperInterface_V2

    print("Master-home support check:")
    print(f"  can={can_name}")
    print(f"  SDK exposes ReqMasterArmMoveToHome: {hasattr(C_PiperInterface_V2, 'ReqMasterArmMoveToHome')}")

    if not hasattr(C_PiperInterface_V2, "ReqMasterArmMoveToHome"):
        raise RuntimeError("installed piper_sdk does not expose ReqMasterArmMoveToHome")

    if dry_run:
        print("  dry-run: skipped CAN connection and firmware query")
        return

    piper = connect_piper(can_name)
    if hasattr(piper, "GetCurrentSDKVersion"):
        print(f"  piper_sdk version: {piper.GetCurrentSDKVersion()}")
    if hasattr(piper, "GetCurrentProtocolVersion"):
        print(f"  protocol version: {piper.GetCurrentProtocolVersion()}")

    version = read_firmware_version(piper, timeout)
    print(f"  firmware: {version}")
    support = is_master_home_firmware(version)
    if support is True:
        print("  firmware check: OK for ReqMasterArmMoveToHome")
    elif support is False:
        raise RuntimeError(
            "firmware is older than S-V1.7-4, which the SDK documents as the "
            "minimum for ReqMasterArmMoveToHome"
        )
    else:
        if allow_unknown_firmware:
            print("  firmware check: UNKNOWN; proceeding because unknown firmware is allowed")
        else:
            raise RuntimeError(
                "could not confirm firmware is S-V1.7-4 or newer. "
                "Set the allow-unknown option only if you have verified this arm firmware manually."
            )


def request_master_home(can_name: str, mode: str, dry_run: bool = False) -> None:
    if mode not in MODES:
        raise ValueError(f"unsupported master-home mode: {mode}")

    code = MODES[mode]
    print(f"Master-home request on {can_name}: ReqMasterArmMoveToHome({code}) [{mode}]")
    if dry_run:
        return

    piper = connect_piper(can_name)
    if not hasattr(piper, "ReqMasterArmMoveToHome"):
        raise RuntimeError(
            "installed piper_sdk does not expose ReqMasterArmMoveToHome; "
            "upgrade piper_sdk or use PARK_METHOD=can_park"
        )
    piper.ReqMasterArmMoveToHome(code)


def read_feedback_zero_error(piper) -> Tuple[float, float, List[float], float, float]:
    joint_msg = piper.GetArmJointMsgs()
    gripper_msg = piper.GetArmGripperMsgs()

    state = joint_msg.joint_state
    joints = [float(getattr(state, f"joint_{idx}")) * RAD_PER_MILLI_DEG for idx in range(1, 7)]
    gripper = float(gripper_msg.gripper_state.grippers_angle) / 1e6
    return max(abs(value) for value in joints), abs(gripper), joints, joint_msg.Hz, gripper_msg.Hz


def wait_until_zero_pose(
    can_name: str,
    timeout: float,
    joint_tolerance: float,
    gripper_tolerance: Optional[float] = None,
    dry_run: bool = False,
) -> None:
    print("Zero-pose verification:")
    print(f"  can={can_name}")
    print(f"  joint_tolerance_rad={joint_tolerance}")
    if gripper_tolerance is None:
        print("  gripper_tolerance_m=disabled")
    else:
        print(f"  gripper_tolerance_m={gripper_tolerance}")

    if dry_run:
        print("  dry-run: skipped feedback polling")
        return

    piper = connect_piper(can_name)
    deadline = time.time() + timeout
    last_report = 0.0
    last_error: Optional[Tuple[float, float, List[float], float, float]] = None

    while time.time() < deadline:
        max_joint_abs, gripper_abs, joints, joint_hz, gripper_hz = read_feedback_zero_error(piper)
        last_error = (max_joint_abs, gripper_abs, joints, joint_hz, gripper_hz)
        live_joint_feedback = joint_hz > 0
        gripper_ok = gripper_tolerance is None or gripper_abs <= gripper_tolerance
        if live_joint_feedback and max_joint_abs <= joint_tolerance and gripper_ok:
            print(
                "  OK: max_joint_abs_rad={:.6f}, gripper_abs_m={:.6f}, "
                "joint_hz={:.2f}, gripper_hz={:.2f}".format(
                    max_joint_abs, gripper_abs, joint_hz, gripper_hz
                )
            )
            return

        now = time.time()
        if now - last_report >= 1.0:
            print(
                "  waiting: max_joint_abs_rad={:.6f}, gripper_abs_m={:.6f}, "
                "joint_hz={:.2f}, gripper_hz={:.2f}".format(
                    max_joint_abs, gripper_abs, joint_hz, gripper_hz
                )
            )
            last_report = now
        time.sleep(0.1)

    if last_error is None:
        raise RuntimeError("did not receive Piper feedback while verifying zero pose")

    max_joint_abs, gripper_abs, joints, joint_hz, gripper_hz = last_error
    raise RuntimeError(
        "zero-pose verification timed out: max_joint_abs_rad={:.6f}, "
        "gripper_abs_m={:.6f}, joint_hz={:.2f}, gripper_hz={:.2f}, "
        "joints_rad={}".format(
            max_joint_abs,
            gripper_abs,
            joint_hz,
            gripper_hz,
            ["{:.6f}".format(value) for value in joints],
        )
    )


def request_master_home_cycle(
    can_name: str,
    wait_seconds: float,
    restore: bool,
    dry_run: bool = False,
    verify_zero: bool = False,
    zero_timeout: float = 8.0,
    joint_tolerance: float = 0.08,
    gripper_tolerance: Optional[float] = None,
) -> None:
    request_master_home(can_name, "both_zero", dry_run)
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for master/slave zero return.")
        if not dry_run:
            time.sleep(wait_seconds)
    if verify_zero:
        wait_until_zero_pose(can_name, zero_timeout, joint_tolerance, gripper_tolerance, dry_run)
    if restore:
        request_master_home(can_name, "restore", dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0", help="CAN interface used by the master-slave pair")
    parser.add_argument("--mode", choices=sorted(MODES), default="both_zero")
    parser.add_argument("--cycle", action="store_true", help="Run both_zero, wait, then restore")
    parser.add_argument("--wait", type=float, default=6.0, help="Wait time used by --cycle")
    parser.add_argument("--check-support", action="store_true", help="Check SDK and firmware support without sending motion-home commands")
    parser.add_argument("--preflight", action="store_true", help="Check SDK and firmware support before sending a request")
    parser.add_argument("--firmware-timeout", type=float, default=3.0, help="Firmware query timeout for --check-support")
    parser.add_argument("--allow-unknown-firmware", action="store_true", help="Allow support check to pass when firmware cannot be parsed")
    parser.add_argument("--no-restore", action="store_true", help="Do not restore master-slave mode after --cycle")
    parser.add_argument("--verify-zero", action="store_true", help="Poll follower feedback and require joints to be near zero before restoring")
    parser.add_argument("--zero-timeout", type=float, default=8.0, help="Timeout for --verify-zero feedback polling")
    parser.add_argument("--joint-tolerance", type=float, default=0.08, help="Max absolute joint error in radians for --verify-zero")
    parser.add_argument("--gripper-tolerance", type=float, default=None, help="Optional max absolute gripper error in meters for --verify-zero")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        if args.check_support:
            check_master_home_support(
                args.can,
                args.firmware_timeout,
                args.dry_run,
                args.allow_unknown_firmware,
            )
            return

        if args.preflight:
            check_master_home_support(
                args.can,
                args.firmware_timeout,
                args.dry_run,
                args.allow_unknown_firmware,
            )

        if args.cycle:
            request_master_home_cycle(
                args.can,
                args.wait,
                not args.no_restore,
                args.dry_run,
                args.verify_zero,
                args.zero_timeout,
                args.joint_tolerance,
                args.gripper_tolerance,
            )
        else:
            request_master_home(args.can, args.mode, args.dry_run)
    except Exception as exc:
        print()
        print("Master-home request failed.")
        print(f"  Error: {exc}")
        print("  Requirements:")
        print("  - Piper firmware V1.7-4 or newer.")
        print("  - The pair is already configured as leader/follower.")
        print("  - The leader and follower are connected on the same CAN bus.")
        print("  - CAN is up at 1000000 bps and the arms are powered.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
