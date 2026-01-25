import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class ImageProcessor:
    """
    Processes scanned habit tracker images.

    Responsible for:
    1. Loading and validating images
    2. Preprocessing (grayscale, binary threshold)
    3. Detecting 4 corner calibration markers
    4. Applying perspective correction to straighten grid

    Attributes:
        config: TrackerConfig instance with detection parameters
        original: Original color image (BGR)
        grayscale: Grayscale version of image
        binary: Binary thresholded image
        corrected: Perspective-corrected image
        corners: Detected corner coordinates [(x,y), ...]
    """

    def __init__(self, config):
        """
        Initialize ImageProcessor with configuration.

        Args:
            config: TrackerConfig instance
        """
        self.config = config
        self.original = None
        self.grayscale = None
        self.binary = None
        self.corrected = None
        self.corners = None

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path and validate.

        Args:
            image_path: Path to the scanned image file

        Returns:
            Loaded image as numpy array (BGR)

        Raises:
            ValueError: If image cannot be loaded or is invalid
            FileNotFoundError: If file doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        self.original = image
        print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
        return image

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert image to grayscale and binary threshold.

        Creates preprocessed versions needed for corner detection.
        Stores results in self.grayscale and self.binary.

        Returns:
            Tuple of (grayscale, binary) images

        Raises:
            ValueError: If original image not loaded yet
        """
        if self.original is None:
            raise ValueError("No image loaded. Call load_image() first.")

        # Convert to grayscale
        self.grayscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold (invert so markers are white on black)
        _, self.binary = cv2.threshold(self.grayscale, 127, 255, cv2.THRESH_BINARY_INV)

        return self.grayscale, self.binary

    def detect_corner_markers(self) -> List[Tuple[int, int]]:
        """
        Detect the 4 corner calibration markers.

        Finds square markers at corners of the habit grid using
        contour detection and filtering by area and aspect ratio.

        Returns:
            List of 4 corner coordinates sorted as:
            [top-left, top-right, bottom-left, bottom-right]

        Raises:
            ValueError: If cannot find exactly 4 valid markers
        """
        if self.binary is None:
            raise ValueError("Image not preprocessed. Call preprocess() first.")

        # Find all contours
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")

        # Filter for marker candidates
        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area
            if self.config.min_area < area < self.config.max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)

                # Check if roughly square
                aspect_ratio = w / h if h > 0 else 0
                if self.config.aspect_ratio_min < aspect_ratio < self.config.aspect_ratio_max:
                    # Use center of bounding box as marker position
                    center_x = x + w // 2
                    center_y = y + h // 2
                    markers.append((center_x, center_y, area))
                    print(f"  Marker candidate at ({center_x}, {center_y}), area={area:.0f}")

        if len(markers) > 4:
            # Take the 4 largest markers
            markers.sort(key=lambda m: m[2], reverse=True)
            markers = markers[:4]
            print(f"Found {len(markers)} candidates, using 4 largest")

        # Remove area from tuples, keep just (x, y)
        marker_points = [(x, y) for x, y, _ in markers]

        # Sort markers into consistent order
        self.corners = self._sort_corners(marker_points)

        return self.corners

    def apply_perspective_correction(self) -> np.ndarray:
        """
        Apply perspective transform to straighten the grid.

        Uses detected corners to warp image into rectangular grid.
        Output dimensions specified in config.

        Returns:
            Perspective-corrected image

        Raises:
            ValueError: If corners not detected yet
        """
        if self.corners is None:
            raise ValueError("Corners not detected. Call detect_corner_markers() first.")

        # Source points (detected corners)
        src_points = np.array(self.corners, dtype=np.float32)

        # Destination points (perfect rectangle)
        dst_points = np.array([
            [0, 0],
            [self.config.output_width - 1, 0],
            [0, self.config.output_height - 1],
            [self.config.output_width - 1, self.config.output_height - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the transform
        self.corrected = cv2.warpPerspective(
            self.original,
            matrix,
            (self.config.output_width, self.config.output_height)
        )

        print(f"\nPerspective correction applied")
        print(f"Output size: {self.config.output_width}x{self.config.output_height}")

        return self.corrected

    def _sort_corners(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort 4 corner points into consistent order.

        Internal helper to ensure corners are ordered as:
        [top-left, top-right, bottom-left, bottom-right]

        Args:
            points: List of 4 (x, y) coordinates in any order

        Returns:
            Sorted list of 4 coordinates
        """
        # Convert to numpy array
        pts = np.array(points, dtype=np.float32)

        # Sort by y-coordinate to get top 2 and bottom 2
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        # Sort each pair by x-coordinate
        top_two = top_two[np.argsort(top_two[:, 0])]
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]

        # Return in order: TL, TR, BL, BR
        sorted_pts = [
            tuple(top_two[0]),
            tuple(top_two[1]),
            tuple(bottom_two[0]),
            tuple(bottom_two[1])
        ]

        print("\nSorted corners:")
        print(f"  Top-left:     {sorted_pts[0]}")
        print(f"  Top-right:    {sorted_pts[1]}")
        print(f"  Bottom-left:  {sorted_pts[2]}")
        print(f"  Bottom-right: {sorted_pts[3]}")

        return sorted_pts

    def get_corrected_image(self) -> Optional[np.ndarray]:
        """
        Retrieve the corrected image.

        Returns:
            Corrected image if available, None otherwise
        """
        return self.corrected

    def visualize_corners(self, output_path: str) -> None:
        """
        Create visualization showing detected corner markers.

        Draws circles and labels at detected corners for debugging.

        Args:
            output_path: Where to save visualization image
        """
        if self.original is None or self.corners is None:
            raise ValueError("No image or corners to visualize")

        vis = self.original.copy()

        # Draw circles at detected corners
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR
        labels = ['TL', 'TR', 'BL', 'BR']

        for i, (x, y) in enumerate(self.corners):
            cv2.circle(vis, (int(x), int(y)), 20, colors[i], -1)
            cv2.putText(vis, labels[i], (int(x) - 30, int(y) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)

        # Draw lines connecting corners
        pts = np.array(self.corners, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts[[0, 1, 3, 2]]], True, (0, 255, 255), 3)

        cv2.imwrite(output_path, vis)
        print(f"Visualization saved to {output_path}")
