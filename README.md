# Habit Tracker Scanner Pipeline

## 1. Project Summary

The Habit Tracker Scanner is a computer vision system that processes scanned images of physical habit tracking worksheets and extracts completion data. The system uses OpenCV for image processing and perspective correction, detecting filled bubbles in a grid format to determine which habits were completed on which days.

**Key Features:**
- Automatic perspective correction using corner calibration markers
- Grid-based bubble detection for habit completion tracking
- Structured data output as pandas DataFrame
- Modular, object-oriented architecture for maintainability

**Current Scope:**
- Single template support (fixed 31-day x 16-habit grid)
- Basic functionality with upright scans
- Bubble detection only (OCR for labels planned for future)
- Graceful failure handling for poor quality scans

**Future Enhancements:**
- OCR for column labels and row notes
- Batch processing capabilities
- Enhanced error handling for edge cases
- Validation and confidence scoring

---

## 2. Workflow

### Processing Pipeline

```
Input: Scanned Image (JPEG/PNG)
    ↓
┌─────────────────────────────────────────────────┐
│ HabitTrackerScanner.scan(image_path)            │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 1: Image Loading & Preprocessing          │
│ [ImageProcessor]                                │
│                                                 │
│ • load_image(path)                              │
│ • Convert to grayscale                          │
│ • Apply binary threshold                        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 2: Corner Marker Detection                │
│ [ImageProcessor]                                │
│                                                 │
│ • detect_corner_markers()                       │
│ • Find 4 calibration squares                    │
│ • Sort corners (TL, TR, BL, BR)                 │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 3: Perspective Correction                 │
│ [ImageProcessor]                                │
│                                                 │
│ • apply_perspective_correction()                │
│ • Warp image to rectangular grid                │
│ • Output: 1700x2200 pixel corrected image       │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 4: Grid Cell Calculation                  │
│ [GridBubbleDetector]                            │
│                                                 │
│ • calculate_grid_cells()                        │
│ • Compute pixel coordinates for 31x16 grid      │
│ • Store cell boundaries (x, y, width, height)   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 5: Bubble Detection                       │
│ [GridBubbleDetector]                            │
│                                                 │
│ • detect_all_bubbles()                          │
│ • For each cell: detect_filled_bubble()         │
│ • Calculate fill ratio (% dark pixels)          │
│ • Apply threshold (>15% = filled)               │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Step 6: Format Results                         │
│ [HabitTrackerScanner]                           │
│                                                 │
│ • Convert detection dict to pandas DataFrame    │
│ • Structure: day | habit_column | is_filled    │
└─────────────────────────────────────────────────┘
    ↓
Output: pandas DataFrame
```

### Data Flow

```
image_path (str)
    ↓
ImageProcessor
    • original_image (BGR)
    • grayscale (single channel)
    • binary (thresholded)
    • corner_points [(x,y), (x,y), (x,y), (x,y)]
    • corrected_image (warped, 1700x2200)
    ↓
GridBubbleDetector
    • grid_cells [31][16] → (x, y, w, h) for each cell
    • detection_results {day: [habit_col_indices]}
    ↓
HabitTrackerScanner
    • results_df (pandas DataFrame)
```

---

## 3. Project Structure

### Class Architecture

```
HabitTrackerScanner (Orchestrator)
│
├── config: TrackerConfig
│   ├── Grid dimensions (31 rows x 16 cols)
│   ├── Grid position (left, top, width, height)
│   ├── Detection parameters (fill_threshold, cell_padding)
│   ├── Output dimensions (1700x2200)
│   └── Corner marker parameters (min/max area, aspect ratio)
│
├── image_processor: ImageProcessor
│   ├── load_image(path)
│   ├── preprocess()
│   ├── detect_corner_markers()
│   ├── apply_perspective_correction()
│   └── _sort_corners()  [private helper]
│
└── grid_bubble_detector: GridBubbleDetector
    ├── calculate_grid_cells()
    ├── detect_filled_bubble(cell_rect)
    └── detect_all_bubbles()
```

### Class Responsibilities

**`TrackerConfig`**
- Centralized configuration management
- All hardcoded parameters in one place
- Grid geometry, detection thresholds, image dimensions

**`ImageProcessor`**
- Image loading and validation
- Preprocessing (grayscale, binary conversion)
- Corner marker detection (4 calibration squares)
- Perspective correction (warp to rectangle)
- Error handling for missing/invalid images

**`GridBubbleDetector`**
- Grid geometry calculation from config
- Individual cell bubble detection
- Full grid scanning
- Results aggregation (day → filled habits)

**`HabitTrackerScanner`**
- Main entry point for users
- Coordinates ImageProcessor and GridBubbleDetector
- Manages data flow between components
- Converts results to pandas DataFrame
- Graceful error handling

### Dependencies

```
opencv-python (cv2)
numpy
pandas
typing (standard library)
pathlib (standard library)
```

### Usage Example (Planned)

```python
from habit_tracker_scanner import HabitTrackerScanner, TrackerConfig

# Initialize scanner with config
config = TrackerConfig.default_31day_grid()
scanner = HabitTrackerScanner(config)

# Process image
results_df = scanner.scan("path/to/scanned_image.jpg")
```

