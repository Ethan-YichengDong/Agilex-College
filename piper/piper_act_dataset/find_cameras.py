#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe V4L2 camera nodes and suggest RGB camera streams for collection."""

import glob
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def video_index(device: str) -> int:
    match = re.search(r"video(\d+)$", device)
    return int(match.group(1)) if match else 10**9


def stable_names(device: str) -> List[str]:
    names = []
    for base in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        for path in glob.glob(os.path.join(base, "*")):
            try:
                if os.path.realpath(path) == device:
                    names.append(path)
            except OSError:
                pass
    return sorted(names)


def usb_speed_label(speed_mbps: Optional[float]) -> str:
    if speed_mbps is None:
        return "unknown"
    if speed_mbps >= 20000:
        return "USB3.2 Gen 2x2"
    if speed_mbps >= 10000:
        return "USB3.2 Gen 2"
    if speed_mbps >= 5000:
        return "USB3.2 Gen 1 / USB3.x"
    if speed_mbps >= 480:
        return "USB2.0 High-Speed"
    if speed_mbps >= 12:
        return "USB1.1 Full-Speed"
    if speed_mbps >= 1.5:
        return "USB1.x Low-Speed"
    return f"{speed_mbps:g} Mb/s"


def usb_info(device: str) -> Dict[str, Any]:
    video_name = os.path.basename(device)
    sysfs_device = os.path.realpath(os.path.join("/sys/class/video4linux", video_name, "device"))
    info: Dict[str, Any] = {
        "sysfs_device": sysfs_device if os.path.exists(sysfs_device) else None,
        "usb_path": None,
        "usb_speed_mbps": None,
        "usb_type": "unknown",
    }

    current = sysfs_device
    for _ in range(8):
        speed_path = os.path.join(current, "speed")
        if os.path.exists(speed_path):
            try:
                with open(speed_path, "r", encoding="utf-8") as handle:
                    speed_mbps = float(handle.read().strip())
            except (OSError, ValueError):
                speed_mbps = None
            info["usb_path"] = current
            info["usb_speed_mbps"] = speed_mbps
            info["usb_type"] = usb_speed_label(speed_mbps)
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return info


def is_realsense_product_name(name: str) -> bool:
    return "RealSense" in name or "Depth_Camera_435" in name


def is_realsense_color_path(name: str) -> bool:
    # On the D435 nodes seen in this lab, the RGB/color stream appears on the
    # USB video interface 1.3, video-index0. Depth/IR streams appeared on 1.0
    # with 424-wide frames or on video-index2/3.
    return "/by-path/" in name and ":1.3-video-index0" in name


def preferred_name(device: str, names: List[str]) -> str:
    by_id = [name for name in names if "/by-id/" in name]
    by_path = [name for name in names if "/by-path/" in name]
    color_paths = [name for name in by_path if is_realsense_color_path(name)]
    if color_paths:
        return color_paths[0]
    # RealSense by-id names contain the product label "Depth Camera 435" even
    # for RGB/color nodes. Prefer by-path in that case so the suggestion does
    # not look like a depth stream.
    if by_path and any(is_realsense_product_name(name) for name in by_id):
        return by_path[0]
    return (by_id or by_path or [device])[0]


def probe(device: str, width: int, height: int) -> Tuple[bool, bool, Optional[Tuple[int, ...]], Optional[float], Optional[int]]:
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


def classify(result: Dict[str, Any], width: int, height: int) -> Tuple[str, str]:
    if not result["ok"]:
        return "unreadable", "OpenCV could not read a frame"

    shape = result["shape"]
    mean = result["mean"]
    max_value = result["max_value"]
    if shape == (height, width, 3):
        if any(is_realsense_color_path(name) for name in result["stable_names"]):
            return "rgb", f"matches requested RGB shape {height}x{width}x3 and RealSense color interface 1.3"
        return "rgb", f"matches requested RGB shape {height}x{width}x3"
    if shape and len(shape) >= 2 and shape[1] != width:
        return "depth_or_ir", f"width {shape[1]} != requested RGB width {width}"
    if mean is not None and max_value == 255 and mean > 220:
        return "depth_or_ir", "very bright stream; likely depth/IR visualization"
    return "unknown", "readable but not an exact RGB match"


def make_result(device: str, width: int, height: int) -> Dict[str, Any]:
    opened, ok, shape, mean, max_value = probe(device, width, height)
    names = stable_names(device)
    result: Dict[str, Any] = {
        "device": device,
        "opened": opened,
        "ok": ok,
        "shape": shape,
        "mean": mean,
        "max_value": max_value,
        "stable_names": names,
        "preferred_name": preferred_name(device, names),
        "usb": usb_info(device),
    }
    stream_type, reason = classify(result, width, height)
    result["stream_type"] = stream_type
    result["reason"] = reason
    return result


def print_result(result: Dict[str, Any]) -> None:
    suffix = ""
    if result["stream_type"] == "rgb":
        suffix = "  <-- RGB candidate"
    elif result["stream_type"] == "depth_or_ir":
        suffix = "  <-- likely depth/IR, skip for RGB"
    elif result["ok"]:
        suffix = "  <-- readable, check manually"
    print(
        "{}: opened={} read={} shape={} mean={} max={} type={} ({}){}".format(
            result["device"],
            result["opened"],
            result["ok"],
            result["shape"],
            result["mean"],
            result["max_value"],
            result["stream_type"],
            result["reason"],
            suffix,
        )
    )
    for name in result["stable_names"]:
        print(f"  stable: {name}")


def print_rgb_usb_situation(rgb: List[Dict[str, Any]]) -> None:
    if not rgb:
        return
    print()
    print("RGB camera USB situation:")
    for item in rgb:
        usb = item["usb"]
        print(f"  {item['device']} -> {usb['usb_type']} ({usb['usb_speed_mbps']} Mb/s)")


def suggest_collection(rgb: List[Dict[str, Any]], width: int, height: int) -> None:
    if len(rgb) < 3:
        print()
        print(f"Only found {len(rgb)} likely RGB stream(s); need 3 for cam_high/cam_left_wrist/cam_right_wrist.")
        if rgb:
            print("Likely RGB streams:")
            for item in rgb:
                print(f"  {item['device']} -> {item['preferred_name']}")
        print_rgb_usb_situation(rgb)
        return

    selected = rgb[:3]
    camera_pairs = [f"{name}={item['preferred_name']}" for name, item in zip(CAMERA_NAMES, selected)]
    print()
    print("Likely RGB streams selected:")
    for name, item in zip(CAMERA_NAMES, selected):
        print(f"  {name}: {item['device']} -> {item['preferred_name']}")

    print()
    print("Suggested collection command:")
    print("  NO_CAMERA=0 \\")
    print(f"  IMAGE_WIDTH={width} IMAGE_HEIGHT={height} \\")
    print(f"  CAMERAS=\"{' '.join(camera_pairs)}\" \\")
    print("  bash piper/piper_act_dataset/collect_act_episode.sh")
    print()
    print("Important: this selects RGB streams automatically, but it cannot know which physical camera is high/left/right.")
    print("If the views are swapped, reorder the three cam_high/cam_left_wrist/cam_right_wrist assignments.")
    print_rgb_usb_situation(selected)


def main() -> None:
    width = int(os.environ.get("IMAGE_WIDTH", "320"))
    height = int(os.environ.get("IMAGE_HEIGHT", "240"))
    results = [make_result(device, width, height) for device in sorted(glob.glob("/dev/video*"), key=video_index)]
    for result in results:
        print_result(result)

    rgb = [result for result in results if result["stream_type"] == "rgb"]
    if rgb:
        suggest_collection(rgb, width, height)
    else:
        print()
        print("No likely RGB camera node found.")


if __name__ == "__main__":
    main()
