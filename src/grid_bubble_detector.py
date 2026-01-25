import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class GridBubbleDetector:
    """
    Detects filled bubbles in the habit tracker grid.

    Responsible for:
    1. Calculating pixel coordinates for each grid cell
    2. Detecting whether each cell/bubble is filled
    3. Returning structured detection results

    Attributes:
        config: TrackerConfig instance with grid and detection parameters
        corrected_image: Perspective-corrected grayscale image
        grid_cells: 2D list [row][col] of (x, y, w, h) cell rectangles
        detection_results: Dictionary mapping day to filled habit columns
    """

    def __init__(self, config):
        """
        Initialize GridBubbleDetector with configuration.

        Args:
            config: TrackerConfig instance
        """
        self.config = config
        self.corrected_image = None
        self.grid_cells = None
        self.detection_results = None

    def set_image(self, corrected_image: np.ndarray) -> None:
        """
        Set the corrected image to analyze.

        Args:
            corrected_image: Perspective-corrected image (grayscale or BGR)
        """
        # Convert to grayscale if needed
        if len(corrected_image.shape) == 3:
            self.corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            self.corrected_image = corrected_image

    def calculate_grid_cells(self) -> List[List[Tuple[int, int, int, int]]]:
        """
        Calculate pixel coordinates for each grid cell.

        Divides the grid into cells based on config dimensions.

        Returns:
            2D list [row][col] of (x, y, width, height) tuples
            for each cell in the grid
        """
        print(f"\n[Step 4] Calculating grid cell positions...")
        print(f"  Grid: {self.config.num_rows} rows x {self.config.num_cols} cols")
        print(f"  Position: ({self.config.grid_left}, {self.config.grid_top})")
        print(f"  Size: {self.config.grid_width} x {self.config.grid_height}")

        cell_width = self.config.grid_width / self.config.num_cols
        cell_height = self.config.grid_height / self.config.num_rows

        print(f"  Cell size: {cell_width:.1f} x {cell_height:.1f} pixels")

        grid_cells = []
        for row in range(self.config.num_rows):
            row_cells = []
            for col in range(self.config.num_cols):
                x = int(self.config.grid_left + col * cell_width)
                y = int(self.config.grid_top + row * cell_height)
                w = int(cell_width)
                h = int(cell_height)
                row_cells.append((x, y, w, h))
            grid_cells.append(row_cells)

        print(f"  Total cells: {self.config.num_rows * self.config.num_cols}")

        self.grid_cells = grid_cells
        return grid_cells

    def detect_filled_bubble(self, cell_rect: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Detect if a single bubble/cell is filled.

        Analyzes the darkness of pixels in the cell region.

        Args:
            cell_rect: (x, y, width, height) of the cell to check

        Returns:
            Tuple of (is_filled: bool, fill_ratio: float)
            fill_ratio is the percentage of dark pixels (0-1)
        """
        if self.corrected_image is None:
            raise ValueError("No image set. Call set_image() first.")

        x, y, w, h = cell_rect

        # Apply padding to sample inner region (avoid grid lines)
        pad_x = int(w * self.config.cell_padding)
        pad_y = int(h * self.config.cell_padding)

        # Extract the inner region of the cell
        roi = self.corrected_image[
            y + pad_y : y + h - pad_y,
            x + pad_x : x + w - pad_x
        ]

        if roi.size == 0:
            return False, 0.0

        # Check if image is already binary (only contains 0 and 255)
        unique_vals = np.unique(roi)
        is_binary = len(unique_vals) <= 2 and all(v in [0, 255] for v in unique_vals)

        if is_binary:
            # Image is already binary from normalization
            # Filled marks are black (0), background is white (255)
            # Count black pixels directly
            dark_pixels = np.sum(roi == 0)
        else:
            # Image is grayscale, apply threshold
            # Threshold to binary (dark pixels = filled marks)
            _, binary = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
            dark_pixels = np.sum(binary == 255)

        # Calculate fill ratio (percentage of dark pixels)
        total_pixels = roi.size
        fill_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0.0

        # Determine if filled based on threshold
        is_filled = fill_ratio > self.config.fill_threshold

        return is_filled, fill_ratio

    def detect_all_bubbles(self) -> Dict[int, List[int]]:
        """
        Detect all filled bubbles in the entire grid.

        Scans every cell and determines which are filled.

        Returns:
            Dictionary mapping day (1-31) to list of filled habit
            column indices (0-indexed)
            Example: {1: [3, 4, 8], 2: [3, 8], ...}
        """
        if self.grid_cells is None:
            raise ValueError("Grid cells not calculated. Call calculate_grid_cells() first.")

        print(f"\n[Step 5] Detecting filled bubbles...")
        print(f"  Fill threshold: {self.config.fill_threshold * 100:.1f}%")

        results = {}
        total_filled = 0

        for row in range(self.config.num_rows):
            day = row + 1  # Days are 1-indexed
            filled_habits = []

            for col in range(self.config.num_cols):
                cell_rect = self.grid_cells[row][col]
                is_filled, fill_ratio = self.detect_filled_bubble(cell_rect)

                if is_filled:
                    filled_habits.append(col)
                    total_filled += 1

            if filled_habits:
                results[day] = filled_habits

        print(f"  Total filled bubbles detected: {total_filled}")
        print(f"  Days with activity: {len(results)}")

        self.detection_results = results

        return results

    def get_results(self) -> Dict[int, List[int]]:
        """
        Retrieve the detection results.

        Returns:
            Dictionary of detection results from last detection run
        """
        return self.detection_results

    def visualize_grid(self, output_path: str, original_image: Optional[np.ndarray] = None) -> None:
        """
        Create visualization showing the expected grid cell locations.

        Draws all grid cells to verify grid positioning is correct.

        Args:
            output_path: Where to save visualization image
            original_image: Optional BGR image to use as base (if None, uses corrected_image)
        """
        if self.grid_cells is None:
            raise ValueError("Grid cells not calculated. Call calculate_grid_cells() first.")

        # Use provided image or corrected image
        if original_image is not None:
            vis = original_image.copy()
        elif self.corrected_image is not None:
            # Convert grayscale to BGR for colored annotations
            if len(self.corrected_image.shape) == 2:
                vis = cv2.cvtColor(self.corrected_image, cv2.COLOR_GRAY2BGR)
            else:
                vis = self.corrected_image.copy()
        else:
            raise ValueError("No image available for visualization")

        print(f"\n[Visualization] Creating grid annotation...")

        # Draw the overall grid boundary
        grid_x1 = self.config.grid_left
        grid_y1 = self.config.grid_top
        grid_x2 = grid_x1 + self.config.grid_width
        grid_y2 = grid_y1 + self.config.grid_height

        # Draw outer boundary in bright color
        cv2.rectangle(vis, (grid_x1, grid_y1), (grid_x2, grid_y2), (0, 255, 255), 3)  # Yellow

        # Draw all grid cells
        for row in range(self.config.num_rows):
            for col in range(self.config.num_cols):
                x, y, w, h = self.grid_cells[row][col]
                # Draw cell boundaries in light gray
                cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # Add row labels (days) on the left
        for row in range(self.config.num_rows):
            x, y, w, h = self.grid_cells[row][0]
            day = row + 1
            # Draw day number to the left of the grid
            label_x = max(5, grid_x1 - 50)
            label_y = y + h // 2 + 5
            cv2.putText(vis, f"{day}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Add column labels (habits) on top
        for col in range(self.config.num_cols):
            x, y, w, h = self.grid_cells[0][col]
            habit = col + 1
            # Draw habit number above the grid
            label_x = x + w // 2 - 8
            label_y = max(20, grid_y1 - 10)
            cv2.putText(vis, f"{habit}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add title and info
        cv2.putText(vis, "Grid Cell Annotation", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(vis, f"{self.config.num_rows} days x {self.config.num_cols} habits",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(output_path, vis)
        print(f"  Grid visualization saved to {output_path}")

    def visualize_detection(self, output_path: str, original_image: Optional[np.ndarray] = None) -> None:
        """
        Create visualization showing detected filled bubbles.

        Draws grid and highlights detected filled cells.

        Args:
            output_path: Where to save visualization image
            original_image: Optional BGR image to use as base (if None, uses corrected_image)
        """
        if self.grid_cells is None or self.detection_results is None:
            raise ValueError("Grid and detection results required. Call detect_all_bubbles() first.")

        # Use provided image or corrected image
        if original_image is not None:
            vis = original_image.copy()
        elif self.corrected_image is not None:
            # Convert grayscale to BGR for colored annotations
            if len(self.corrected_image.shape) == 2:
                vis = cv2.cvtColor(self.corrected_image, cv2.COLOR_GRAY2BGR)
            else:
                vis = self.corrected_image.copy()
        else:
            raise ValueError("No image available for visualization")

        print(f"\n[Visualization] Creating annotated image...")

        # Draw all grid cells (light gray)
        for row in range(self.config.num_rows):
            for col in range(self.config.num_cols):
                x, y, w, h = self.grid_cells[row][col]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 200, 200), 1)

        # Highlight detected filled cells
        for day, habit_cols in self.detection_results.items():
            row = day - 1  # Convert back to 0-indexed
            for col in habit_cols:
                x, y, w, h = self.grid_cells[row][col]

                # Draw green rectangle around detected filled cell
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Add checkmark
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.drawMarker(vis, (center_x, center_y), (0, 200, 0),
                              cv2.MARKER_TILTED_CROSS, 15, 2)

        # Add legend
        legend_y = 50
        cv2.putText(vis, "Green = Detected Filled", (50, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        total_bubbles = sum(len(cols) for cols in self.detection_results.values())
        cv2.putText(vis, f"Total: {total_bubbles} bubbles",
                   (50, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imwrite(output_path, vis)
        print(f"  Visualization saved to {output_path}")

    def create_fill_ratio_heatmap(self, output_path: str) -> None:
        """
        Create heatmap showing fill ratios for all cells.

        Useful for debugging and tuning the fill_threshold parameter.

        Args:
            output_path: Where to save heatmap image
        """
        if self.grid_cells is None:
            raise ValueError("Grid cells not calculated. Call calculate_grid_cells() first.")
        if self.corrected_image is None:
            raise ValueError("No image set. Call set_image() first.")

        print(f"\n[Debug Visualization] Creating fill ratio heatmap...")

        # Create a blank image for the heatmap
        h = self.config.num_rows * 30
        w = self.config.num_cols * 30
        heatmap = np.ones((h, w, 3), dtype=np.uint8) * 255

        cell_h = h // self.config.num_rows
        cell_w = w // self.config.num_cols

        for row in range(self.config.num_rows):
            for col in range(self.config.num_cols):
                cell_rect = self.grid_cells[row][col]
                _, fill_ratio = self.detect_filled_bubble(cell_rect)

                # Map fill ratio to color (blue = empty, red = filled)
                # Using HSV color space for better visualization
                hue = int((1 - fill_ratio) * 120)  # 120 = blue, 0 = red
                sat = 255
                val = 255
                color_hsv = np.uint8([[[hue, sat, val]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]

                # Draw cell
                y1 = row * cell_h
                x1 = col * cell_w
                cv2.rectangle(heatmap, (x1, y1), (x1 + cell_w, y1 + cell_h),
                             color_bgr.tolist(), -1)

                # Add text showing ratio
                text = f"{fill_ratio:.2f}"
                font_scale = 0.3
                cv2.putText(heatmap, text, (x1 + 2, y1 + cell_h - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        cv2.imwrite(output_path, heatmap)
        print(f"  Heatmap saved to {output_path}")
