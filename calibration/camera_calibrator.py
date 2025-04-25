#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


class CameraCalibrator:
    def __init__(
        self,
        images_folder="chessboard_images",
        output_file="calibration/camera_calibration.npz",
        board_size=(7, 6),
        square_size=1.0,
    ):
        """
        Initializes the camera calibrator

        Args:
            images_folder: folder with chessboard images
            output_file: file to save calibration results
            board_size: chessboard size (width, height) of inner corners
            square_size: size of the square on chessboard (arbitrary units)
        """
        self.images_folder = images_folder

        # Ensure the calibration directory exists
        if not os.path.exists("calibration"):
            os.makedirs("calibration")

        self.output_file = output_file
        self.board_size = board_size
        self.square_size = square_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None

    def find_best_board_size(self, show_results=False):
        """Determines the best chessboard size for the given image set"""
        images = self._get_images()
        if not images:
            return None

        # Various chessboard size options
        board_sizes = [
            (9, 6),
            (8, 6),
            (7, 6),
            (6, 5),
            (7, 5),
            (8, 5),
            (10, 7),
            (11, 8),
            (7, 7),
            (5, 5),
            (4, 4),
            (5, 4),
            (6, 4),
            (8, 7),
            (9, 7),
            (6, 6),
        ]

        results = {}
        best_size = None
        best_count = 0

        for size in board_sizes:
            count = 0
            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                ret, _ = cv2.findChessboardCorners(gray, size, flags)
                if ret:
                    count += 1

            print(f"Board size {size}: found in {count} of {len(images)} images")
            results[size] = count

            if count > best_count:
                best_count = count
                best_size = size

        if best_size:
            print(
                f"\nBest chessboard size: {best_size} - found in {best_count} of {len(images)} images"
            )

            # Update board size
            self.board_size = best_size

        return best_size

    def calibrate(self, debug_mode=False):
        """Calibrates the camera and returns the camera matrix"""
        images = self._get_images()
        if not images:
            print(f"No images found in {self.images_folder}")
            return None, None

        # Prepare object points
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0 : self.board_size[0], 0 : self.board_size[1]].T.reshape(-1, 2)
            * self.square_size
        )

        # Arrays to store points
        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image plane

        img_size = None
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

        success_count = 0

        # Process all images
        for i, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

            if ret:
                objpoints.append(objp)

                # Refine corner positions
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                success_count += 1
                print(f"Image {i+1}/{len(images)}: chessboard found successfully")

                # Display detected corners
                if debug_mode:
                    img_copy = img.copy()
                    cv2.drawChessboardCorners(img_copy, self.board_size, corners2, ret)

                    # Scale if needed
                    h, w = img_copy.shape[:2]
                    if h > 800 or w > 1200:
                        scale = min(800 / h, 1200 / w)
                        img_copy = cv2.resize(img_copy, (0, 0), fx=scale, fy=scale)

                    cv2.imshow("Chess board", img_copy)
                    key = cv2.waitKey(200)  # 200ms to view
                    if key == 27:  # ESC to exit
                        cv2.destroyAllWindows()
                        break
            else:
                print(f"Image {i+1}/{len(images)}: chessboard NOT found")

        # Close display windows
        if debug_mode:
            cv2.destroyAllWindows()

        # Perform calibration if we have successful images
        if success_count > 0:
            print(f"Performing calibration based on {success_count} images...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )

            # Optimize camera matrix
            W, H = img_size
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (W, H), 1, (W, H)
            )

            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    objpoints[i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                    imgpoints2
                )
                total_error += error

            self.calibration_error = total_error / len(objpoints)

            print(f"Calibration completed successfully! Used {success_count} images.")
            print(f"Average reprojection error: {self.calibration_error}")
            print(f"Camera matrix:\n{newcameramtx}")

            self.camera_matrix = newcameramtx
            self.dist_coeffs = dist

            return newcameramtx, dist
        else:
            print("Could not find chessboard in any image.")
            return None, None

    def save_calibration(self):
        """Saves calibration results to file"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            np.savez(
                self.output_file,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                error=self.calibration_error,
                board_size=np.array(self.board_size),
            )
            print(f"Calibration results saved to file {self.output_file}")
            return True
        
        return False

    def load_calibration(self):
        """Loads calibration results from file"""
        # List of paths to search for calibration file
        search_paths = [
            self.output_file,
            "camera_calibration.npz",
            os.path.join("calibration", "camera_calibration.npz"),
            os.path.join(os.path.dirname(__file__), "camera_calibration.npz"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                try:
                    data = np.load(path)
                    self.camera_matrix = data["camera_matrix"]
                    self.dist_coeffs = data["dist_coeffs"]
                    if "error" in data:
                        self.calibration_error = data["error"]
                    if "board_size" in data:
                        self.board_size = tuple(data["board_size"])

                    print(
                        f"Loaded existing camera calibration from {path}:\n{self.camera_matrix}"
                    )
                    return True
                except Exception as e:
                    print(f"Error reading file {path}: {e}")

        print("Could not find calibration file.")
        return False

    def show_test_result(self):
        """Shows calibration results on one of the images"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("No calibration data. Please calibrate first.")
            return

        images = self._get_images()
        if not images:
            print("No images found for testing")
            return

        # Find a suitable image
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Check if chessboard is found
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, _ = cv2.findChessboardCorners(gray, self.board_size, None)
            if ret:
                print(f"Using image: {os.path.basename(img_path)}")
                break
        else:
            print("Could not find a suitable image with a chessboard")
            return

        h, w = img.shape[:2]

        # Fix distortion
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            img, self.camera_matrix, self.dist_coeffs, None, newcameramtx
        )

        # Crop unnecessary areas
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            undistorted_cropped = undistorted[y : y + h, x : x + w]
            print(f"Crop: x={x}, y={y}, w={w}, h={h}")
        else:
            undistorted_cropped = undistorted
            print("Cannot crop image, ROI is invalid")

        # Resize images
        scale = 0.5
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        undistorted_small = cv2.resize(
            undistorted, (img_small.shape[1], img_small.shape[0])
        )

        # Create result for display
        if (
            "undistorted_cropped" in locals()
            and undistorted_cropped.shape != undistorted.shape
        ):
            # Scale to same height
            h_small = img_small.shape[0]
            scale_factor = h_small / undistorted_cropped.shape[0]
            cropped_small = cv2.resize(
                undistorted_cropped, (0, 0), fx=scale_factor, fy=scale_factor
            )

            # Create final image
            result_width = (
                img_small.shape[1]
                + undistorted_small.shape[1]
                + cropped_small.shape[1]
                + 20
            )
            result = np.zeros((img_small.shape[0], result_width, 3), dtype=np.uint8)

            # Add images
            offset = 0
            result[: img_small.shape[0], offset : offset + img_small.shape[1]] = (
                img_small
            )

            offset += img_small.shape[1] + 10
            result[
                : undistorted_small.shape[0],
                offset : offset + undistorted_small.shape[1],
            ] = undistorted_small

            offset += undistorted_small.shape[1] + 10
            result[
                : cropped_small.shape[0], offset : offset + cropped_small.shape[1]
            ] = cropped_small

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, "Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(
                result,
                "Fixed",
                (img_small.shape[1] + 20, 30),
                font,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                result,
                "Cropped",
                (img_small.shape[1] + undistorted_small.shape[1] + 30, 30),
                font,
                1,
                (0, 255, 0),
                2,
            )
        else:
            # Only two images
            result = np.zeros(
                (img_small.shape[0], img_small.shape[1] * 2 + 10, 3), dtype=np.uint8
            )
            result[:, : img_small.shape[1]] = img_small
            result[:, img_small.shape[1] + 10 :] = undistorted_small

            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, "Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(
                result,
                "Fixed",
                (img_small.shape[1] + 20, 30),
                font,
                1,
                (0, 255, 0),
                2,
            )

        # Show and save result
        cv2.imshow("Calibration result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        result_file_calibration = "calibration/calibration_result.jpg"

        os.makedirs("calibration", exist_ok=True)
        cv2.imwrite(result_file_calibration, result)

        print(f"Result saved to {result_file_calibration}")

    def get_camera_matrix(
        self, force_recalibration=False, debug_mode=False, check_board_size=False
    ):
        """
        Gets the camera matrix - loads from file or performs calibration

        Args:
            force_recalibration: force recalibration
            debug_mode: whether to show calibration process
            check_board_size: determine the best board size

        Returns:
            camera_matrix: camera intrinsic parameters matrix
        """
        # Check for saved calibration
        if not force_recalibration and self.load_calibration():
            return self.camera_matrix

        # Determine optimal board size if needed
        if check_board_size:
            self.find_best_board_size()

        # Camera calibration
        self.calibrate(debug_mode)

        # Save results
        if self.camera_matrix is not None:
            self.save_calibration()

        return self.camera_matrix

    def _get_images(self):
        """Gets a list of images in the folder"""
        images = glob.glob(os.path.join(self.images_folder, "*.jpg"))
        if not images:
            for ext in ["*.png", "*.jpeg", "*.bmp"]:
                images = glob.glob(os.path.join(self.images_folder, ext))
                if images:
                    break
        return images


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Camera calibration using a chessboard"
    )

    parser.add_argument(
        "--folder",
        default="chessboard_images",
        help="Folder with chessboard images",
    )
    parser.add_argument(
        "--output",
        default="calibration/camera_calibration.npz",
        help="File to save calibration results",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Chessboard size in WxH format (e.g., 7x6)",
    )
    parser.add_argument(
        "--find-size",
        action="store_true",
        help="Determine optimal chessboard size",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode - show all steps"
    )
    parser.add_argument("--force", action="store_true", help="Force recalibration")
    parser.add_argument("--test", action="store_true", help="Show calibration result")

    args = parser.parse_args()

    # Determine board size
    board_size = (7, 6)  # Default
    if args.size:
        try:
            w, h = map(int, args.size.split("x"))
            board_size = (w, h)
            print(f"Board size set to: {board_size}")
        except:
            print("Invalid board size format. Using default value.")

    # Create calibrator
    calibrator = CameraCalibrator(
        images_folder=args.folder, output_file=args.output, board_size=board_size
    )

    # Determine action
    if args.find_size:
        best_size = calibrator.find_best_board_size()
        if best_size and args.debug:
            # If we found optimal size and debug mode is on, calibrate immediately
            calibrator.board_size = best_size
            print(f"Continuing calibration with board size {best_size}...")
            calibrator.get_camera_matrix(
                force_recalibration=args.force, debug_mode=args.debug
            )
            if args.test:
                calibrator.show_test_result()
    elif args.test and os.path.exists(args.output):
        calibrator.load_calibration()
        calibrator.show_test_result()
    else:
        # Get camera matrix
        K = calibrator.get_camera_matrix(
            force_recalibration=args.force,
            debug_mode=args.debug,
            check_board_size=args.find_size,
        )

        if K is not None and args.test:
            calibrator.show_test_result()


if __name__ == "__main__":
    main()
