# Habit Tracker Scanner

A computer vision pipeline that reads a printed habit-tracking worksheet from a photo and identifies which bubbles were filled in.

The worksheet is a grid—31 rows (days of the month) by 15 columns (habits)—with four ArUco fiducial markers printed at the corners for reliable alignment. A user fills in bubbles by hand, photographs the sheet, and runs this scanner to digitize the results.

## Usage

```bash

python -m src.scanner ./input/version_2_filled3.jpg
```

### Requirements

```
pip install -r requirements.txt
```

Core dependencies: OpenCV, NumPy, pandas, Matplotlib, Pillow.

## Processing Pipeline

The scanner executes six sequential steps. Each step saves a debug image to `output/` so you can visually verify every stage.

1. Load & Preprocess — Load image, convert to grayscale and binary
2. Detect ArUco Corner Markers — Find four markers (IDs 0–3) to identify worksheet corners
<div align="center">
	<img src="https://github.com/trngt/habit-tracker-ocr/blob/main/figures/1_detected_markers.jpg" width="50%"/>
  <p>Figure 1. Detection of ArUco Markers</p>
</div>
3. Perspective Correction — Warp image into a canonical rectangle via homography transform
4. Histogram Analysis & Normalization — Apply CLAHE and binarize to black-and-white
5. Grid Cell Calculation — Divide grid region into num_rows × num_cols cells
6. Bubble Detection — Crop each cell, compute fill ratio, mark filled cells
<div align="center">
	<img src="https://github.com/trngt/habit-tracker-ocr/blob/main/figures/5_bubble_detection.jpg" width="50%"/>
  <p>Figure 2. Bubble detection</p>
</div>
7. Column Header OCR (in progress) — Read habit names above each column via OCR
8. Write to csv (in progress)
