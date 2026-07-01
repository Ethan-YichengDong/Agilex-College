#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import h5py


def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{name}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", help="Path to episode_N.hdf5")
    args = parser.parse_args()
    with h5py.File(args.episode, "r") as root:
        print("attrs:")
        for key, value in root.attrs.items():
            print(f"  {key}: {value}")
        print("datasets:")
        root.visititems(visit)


if __name__ == "__main__":
    main()
