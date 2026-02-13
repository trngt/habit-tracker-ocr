# Habit Tracker Scanner

A computer vision pipeline that reads a printed habit-tracking worksheet from a photo and identifies which bubbles were filled in.

The worksheet is a grid — 31 rows (days of the month) by 15 columns (habits) — with four ArUco fiducial markers printed at the corners for reliable alignment. A user fills in bubbles by hand, photographs the sheet, and runs this scanner to digitize the results.

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

### Step 1 — Load & Preprocess

Loads the input image and converts it to grayscale and binary (inverted) representations needed by downstream steps.

**Output:** raw image loaded into memory.

### Step 2 — Detect ArUco Corner Markers

Detects four ArUco markers (IDs 0–3, from a `DICT_4X4_50` dictionary) and maps them to the four corners of the worksheet:

| Marker ID | Corner |
|---|---|
| 0 | Top-left |
| 1 | Top-right |
| 2 | Bottom-left |
| 3 | Bottom-right |

The center of each detected marker becomes a reference point for perspective correction. The scan fails with a clear error if any marker is missing.

**Output:** `1_detected_markers.jpg` — original image annotated with detected marker positions and the quadrilateral they form.

### Step 3 — Perspective Correction

Computes a perspective (homography) transform from the four detected marker centers to a perfect rectangle of configurable size (`output_width` × `output_height`, default 1700×2200 px). The original image is warped into this canonical view, removing camera tilt and rotation.

**Output:** `2_corrected_grid.jpg`, `2_corrected_grid_gray.jpg`

### Step 3.5 — Histogram Analysis & Normalization

Analyzes the pixel intensity distribution of the corrected image and binarizes it to pure black-and-white:

1. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to reduce shadow and lighting variation.
2. Determines a binarization threshold — either a manually supplied value or one computed automatically at 50% of Otsu's optimal threshold.
3. Thresholds the image so filled marks become black (0) and paper/grid lines become white (255).

**Output:** `3_histogram.png` (intensity distribution with threshold marked), `3_normalized_image.jpg`

### Step 4 — Grid Cell Calculation

Divides the known grid region into `num_rows × num_cols` cells using the pixel offsets and dimensions from the config (`grid_left`, `grid_top`, `grid_width`, `grid_height`). Each cell is stored as an `(x, y, width, height)` rectangle.

**Output:** `4_grid_annotation.jpg` — corrected image overlaid with the computed grid, row/column labels, and the column header region.

### Step 5 — Bubble Detection

For each cell in the grid:

1. Crops the inner region of the cell (shrunk by `cell_padding` on each side to avoid grid lines).
2. Counts dark (black) pixels in the cropped region.
3. Computes a **fill ratio** = dark pixels / total pixels.
4. Marks the cell as filled if the fill ratio exceeds `fill_threshold`.

Results are returned as a dictionary mapping each day (1–31) to a list of filled column indices.

**Output:** `5_bubble_detection.jpg` (grid with green highlights on detected bubbles), `5_fill_ratios_heatmap.jpg` (color-coded heatmap of fill ratios for threshold tuning).

### Step 6 — Column Header OCR *(work in progress)*

Reads the habit names printed above each column using OCR. The header region is extracted, sliced per-column, rotated 90° (text is printed vertically), and passed to an OCR engine. This step is currently commented out in the main pipeline.
