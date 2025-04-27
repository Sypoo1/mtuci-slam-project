#!/usr/bin/env python3

import argparse
import os

import numpy as np


def get_camera_matrix(calibration_file="calibration/camera_calibration.npz"):
    """
    Loads camera matrix from calibration file

    Args:
        calibration_file: path to calibration file

    Returns:
        camera_matrix: intrinsic camera parameters matrix
    """
    # Check all possible file paths
    file_paths = [
        calibration_file,  # Given path
        os.path.join(
            "calibration", os.path.basename(calibration_file)
        ),  # In calibration/ folder
        os.path.join(
            os.path.dirname(__file__), os.path.basename(calibration_file)
        ),  # Next to current file
    ]

    # Find existing file
    actual_path = None
    for file_path in file_paths:
        if os.path.exists(file_path):
            actual_path = file_path
            break

    if actual_path is None:
        print(f"Calibration file not found in any of these paths:")
        for path in file_paths:
            print(f"  - {path}")
        # Return approximate matrix
        W, H = 1280, 720
        F = 900  # approximate focal length
        K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]], dtype=np.float32)
        return K

    # Load calibration
    try:
        data = np.load(actual_path)
        K = data["camera_matrix"]
        dist = data["dist_coeffs"]

        if "error" in data:
            error = data["error"]
            print(f"Calibration error: {error}")

        if "board_size" in data:
            board_size = tuple(data["board_size"])
            print(f"Chessboard size: {board_size}")

        print(f"Loaded camera matrix from {actual_path}:")
        print(K)

        return K
    except Exception as e:
        print(f"Error reading calibration file {actual_path}: {e}")
        # Return approximate matrix
        W, H = 1280, 720
        F = 900  # approximate focal length
        K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]], dtype=np.float32)
        return K


# This will allow us to run this module directly to show the camera matrix
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
