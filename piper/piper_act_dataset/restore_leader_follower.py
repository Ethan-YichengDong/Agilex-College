#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore Piper leader-follower teaching mode.

Leader/input arms use MasterSlaveConfig(0xFA, 0, 0, 0).
Follower/output arms use MasterSlaveConfig(0xFC, 0, 0, 0).
"""

import argparse
import time
from typing import Iterable, Tuple


LEADER_CONFIG = 0xFA
FOLLOWER_CONFIG = 0xFC


def connect(can_name: str):
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.1)
    return piper


def configure_arm(can_name: str, role: str, dry_run: bool = False) -> None:
    if role == "leader":
        config = LEADER_CONFIG
    elif role == "follower":
        config = FOLLOWER_CONFIG
    else:
        raise ValueError(f"unsupported role: {role}")

    print(f"Restoring {role} mode on {can_name}: MasterSlaveConfig(0x{config:02X}, 0, 0, 0)")
    print("  Broadcast warning: every Piper arm listening on this CAN interface can receive this role command.")
    if dry_run:
        return

    piper = connect(can_name)
    piper.MasterSlaveConfig(config, 0, 0, 0)
    time.sleep(0.2)


def restore_pairs(pairs: Iterable[Tuple[str, str]], dry_run: bool = False) -> None:
    for role, can_name in pairs:
        if can_name:
            configure_arm(can_name, role, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-can", action="append", default=[], help="Leader/input arm CAN interface. May be repeated.")
    parser.add_argument("--follower-can", action="append", default=[], help="Follower/output arm CAN interface. May be repeated.")
    parser.add_argument(
        "--restore-order",
        choices=["leader_first", "follower_first"],
        default="leader_first",
        help="Order used when multiple role commands are restored.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.leader_can and not args.follower_can:
        raise SystemExit("configure at least one --leader-can or --follower-can")

    shared = sorted(set(args.leader_can).intersection(args.follower_can))
    if shared:
        shared_list = ", ".join(shared)
        raise SystemExit(
            "refusing to configure leader and follower roles on the same CAN interface "
            f"({shared_list}) in one run. Power/connect only one physical arm and run "
            "one role at a time, or use separate CAN interfaces / validated CAN-ID offsets."
        )

    if args.restore_order == "leader_first":
        pairs = [("leader", can_name) for can_name in args.leader_can]
        pairs.extend(("follower", can_name) for can_name in args.follower_can)
    else:
        pairs = [("follower", can_name) for can_name in args.follower_can]
        pairs.extend(("leader", can_name) for can_name in args.leader_can)
    try:
        restore_pairs(pairs, args.dry_run)
    except Exception as exc:
        print()
        print("Failed to restore leader-follower mode.")
        print(f"  Error: {exc}")
        print("  Check that each CAN interface is correct and that the arm is powered.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
