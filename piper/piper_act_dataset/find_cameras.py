#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe V4L2 camera nodes and show which ones OpenCV can read."""

import glob
import os
import time

import cv2


def stable_names(device: str) -> list[str]:
    names = []
    for base in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        for path in glob.glob(os.path.join(base, "*")):
            try:
                if os.path.realpath(path) == device:
                    names.append(path)
            except OSError:
                pass
    return sorted(names)


def probe(device: str, width: int, height: int) -> tuple[bool, bool, tuple[int, ...] | None, float | None, int | None]:
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    opened = cap.isOpened()
    ok = False
    shape = None
    mean = None
    max_value = None
    if opened:
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = tuple(frame.shape)
                mean = float(frame.mean())
                max_value = int(frame.max())
                break
            time.sleep(0.05)
    cap.release()
    return opened, ok, shape, mean, max_value


def main() -> None:
    width = int(os.environ.get("IMAGE_WIDTH", "320"))
    height = int(os.environ.get("IMAGE_HEIGHT", "240"))
    usable = []
    shapes = {}
    for device in sorted(glob.glob("/dev/video*")):
        opened, ok, shape, mean, max_value = probe(device, width, height)
        shapes[device] = shape
        suffix = ""
        if ok:
            usable.append(device)
            suffix = "  <-- usable"
        print(f"{device}: opened={opened} read={ok} shape={shape} mean={mean} max={max_value}{suffix}")
        for name in stable_names(device):
            print(f"  stable: {name}")

    if usable:
        preferred = next((device for device in usable if shapes[device] == (height, width, 3)), usable[0])
        print()
        print("Suggested collection command:")
        print(f"  CAMERA_DEVICE={preferred} bash piper/piper_act_dataset/collect_act_episode.sh")
    else:
        print()
        print("No readable OpenCV camera node found.")


if __name__ == "__main__":
    main()
