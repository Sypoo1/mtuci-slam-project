# Camera Calibration

Module for camera calibration using a chessboard pattern for use in SLAM systems.

## Quick Calibration (all in one command)

For the fastest calibration execution:

```bash
python -m calibration.camera_calibrator --find-size --debug --force --test
```

This command automatically:
1. Determines the optimal chessboard size
2. Performs calibration with that size
3. Shows calibration results (visually and in console)
4. Saves the camera matrix to files `calibration/camera_calibration.npz` and `camera_calibration.npz`

## Step-by-Step Calibration (for better control)

### 1. Preparing Images

1. Take 15-20 photos of a chessboard from different angles
2. Place them in the `chessboard_images/` folder
3. Formats: `.jpg`, `.png`, `.jpeg`, `.bmp`

### 2. Determining Chessboard Size

Determine the optimal chessboard size:

```bash
python -m calibration.camera_calibrator --find-size
```

### 3. Performing Calibration

Use the found size for calibration (replace 7x6 with the found size):

```bash
python -m calibration.camera_calibrator --size=7x6 --debug --force
```

### 4. Checking Results

Verify the calibration results:

```bash
python -m calibration.camera_calibrator --test
```

## Troubleshooting

If empty black windows appear during calibration in debug mode:
- This is a display issue with OpenCV, the calibration results will still be saved
- You can run without the `--debug` flag to avoid showing windows

If the chessboard cannot be found in images:
- Make sure the chessboard is clearly visible in the photos
- Check lighting and contrast
- Try a different board size (e.g. `--size=8x7`)

## Using the Camera Matrix

To get the camera matrix in your code:

```python
from calibration import get_camera_matrix

# Get the camera matrix and distortion coefficients
K, dist = get_camera_matrix()

# Now you can use K in your SLAM system
```

## Other Options

- `--folder=path` - folder with images (default `chessboard_images`)
- `--output=path` - path to save calibration (default `calibration/camera_calibration.npz`)
- `--force` - force recalibration
- `--debug` - show the chessboard detection process
- `--test` - display calibration results

## Viewing Current Calibration Information

To display information about the current camera matrix without performing calibration:

```bash
# Use this command (avoids warnings)
python -m calibration.show_matrix

# Alternative (may show import warning)
python -m calibration.get_camera_matrix
```

## Notes

- Calibration results are saved in two locations: in `calibration/camera_calibration.npz` and in the project root `camera_calibration.npz` for compatibility
- The visual calibration result is also saved to the file `calibration_result.jpg`