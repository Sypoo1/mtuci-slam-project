#!/usr/bin/env python3
"""
Simple command line tool to show the camera matrix
Run this directly with:
python -m calibration.show_matrix
"""

import argparse

from calibration import get_camera_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Get camera matrix from calibration file"
    )
    parser.add_argument(
        "--file", default="camera_calibration.npz", help="Path to calibration file"
    )
    args = parser.parse_args()

    K = get_camera_matrix(args.file)

    print("\nCamera matrix K:")
    print(K)



if __name__ == "__main__":
    main()
