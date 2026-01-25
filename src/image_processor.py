import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib


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

    def detect_aruco_markers(self) -> List[Tuple[int, int]]:
        """
        Detect the 4 ArUco corner calibration markers.

        Finds ArUco markers with IDs 0-3 and maps them to corners.
        ID mapping: 0=TL, 1=TR, 2=BL, 3=BR

        Returns:
            List of 4 corner coordinates sorted as:
            [top-left, top-right, bottom-left, bottom-right]

        Raises:
            ValueError: If cannot find all 4 required markers (IDs 0,1,2,3)
        """
        if self.grayscale is None:
            raise ValueError("Image not preprocessed. Call preprocess() first.")

        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.config.aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Detect markers
        corners_detected, ids, rejected = detector.detectMarkers(self.grayscale)

        print(f"\n[ArUco Detection]")
        if ids is not None:
            print(f"  Found {len(ids)} ArUco marker(s): {ids.flatten().tolist()}")
        else:
            raise ValueError("No ArUco markers detected in image")

        # Flatten IDs array
        ids = ids.flatten()

        # Check we have all required markers (0, 1, 2, 3)
        required_ids = {0, 1, 2, 3}
        found_ids = set(ids.tolist())
        missing_ids = required_ids - found_ids

        if missing_ids:
            raise ValueError(f"Missing required ArUco markers: {sorted(missing_ids)}")

        # Map marker IDs to corner positions
        # Each marker's corners are in order: TL, TR, BR, BL (of the marker itself)
        # We use the center of each marker as the corner point
        marker_centers = {}

        for i, marker_id in enumerate(ids):
            if marker_id in required_ids:
                # Get the 4 corners of this marker
                marker_corners = corners_detected[i][0]
                # Calculate center point
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                marker_centers[marker_id] = (center_x, center_y)
                print(f"  Marker ID {marker_id}: center at ({center_x}, {center_y})")

        # Build ordered corner list based on marker IDs
        # ID 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
        self.corners = [
            marker_centers[0],  # top-left
            marker_centers[1],  # top-right
            marker_centers[2],  # bottom-left
            marker_centers[3]   # bottom-right
        ]

        print(f"\nOrdered corners:")
        print(f"  Top-left (ID 0):     {self.corners[0]}")
        print(f"  Top-right (ID 1):    {self.corners[1]}")
        print(f"  Bottom-left (ID 2):  {self.corners[2]}")
        print(f"  Bottom-right (ID 3): {self.corners[3]}")

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

    def get_corrected_image(self) -> Optional[np.ndarray]:
        """
        Retrieve the corrected image.

        Returns:
            Corrected image if available, None otherwise
        """
        return self.corrected

    def visualize_corners(self, output_path: str) -> None:
        """
        Create visualization showing detected ArUco corner markers.

        Draws circles and labels at detected corners with ArUco IDs for debugging.

        Args:
            output_path: Where to save visualization image
        """
        if self.original is None or self.corners is None:
            raise ValueError("No image or corners to visualize")

        vis = self.original.copy()

        # Draw circles at detected corners with ArUco IDs
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR
        labels = ['ID:0 (TL)', 'ID:1 (TR)', 'ID:2 (BL)', 'ID:3 (BR)']

        for i, (x, y) in enumerate(self.corners):
            cv2.circle(vis, (int(x), int(y)), 20, colors[i], -1)
            cv2.putText(vis, labels[i], (int(x) - 50, int(y) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

        # Draw lines connecting corners
        pts = np.array(self.corners, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts[[0, 1, 3, 2]]], True, (0, 255, 255), 3)

        cv2.imwrite(output_path, vis)
        print(f"Visualization saved to {output_path}")

    def analyze_histogram(self, image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze histogram of pixel intensity values.

        Args:
            image: Optional grayscale image to analyze (default: uses corrected image)

        Returns:
            Tuple of (histogram values, bin edges)

        Raises:
            ValueError: If no image available
        """
        if image is None:
            if self.corrected is None:
                raise ValueError("No corrected image available. Call apply_perspective_correction() first.")
            # Convert corrected image to grayscale if needed
            if len(self.corrected.shape) == 3:
                image = cv2.cvtColor(self.corrected, cv2.COLOR_BGR2GRAY)
            else:
                image = self.corrected

        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()

        return hist, np.arange(256)

    def visualize_histogram(self, output_path: str, threshold: Optional[int] = None) -> None:
        """
        Create histogram visualization with matplotlib.

        Args:
            output_path: Where to save histogram figure
            threshold: Optional threshold value to mark on histogram
        """
        # Use matplotlib Agg backend to avoid display issues
        matplotlib.use('Agg')

        hist, bins = self.analyze_histogram()

        print(f"\n[Histogram Analysis]")
        print(f"  Pixel intensity range: 0-255")
        print(f"  Mean intensity: {np.average(bins, weights=hist):.1f}")
        print(f"  Median intensity: {np.median(np.repeat(bins, hist.astype(int))):.1f}")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram
        ax.bar(bins, hist, width=1.0, color='blue', alpha=0.7, label='Pixel Distribution')
        ax.set_xlabel('Pixel Intensity (0=Black, 255=White)', fontsize=12)
        ax.set_ylabel('Frequency (Number of Pixels)', fontsize=12)
        ax.set_title('Pixel Intensity Histogram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark threshold if provided
        if threshold is not None:
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold}')
            ax.legend()

        # Add statistics text
        stats_text = f"Mean: {np.average(bins, weights=hist):.1f}\n"
        stats_text += f"Median: {np.median(np.repeat(bins, hist.astype(int))):.1f}"
        ax.text(0.98, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"  Histogram saved to {output_path}")

    def compute_adaptive_threshold(self) -> int:
        """
        Compute optimal threshold using Otsu's method.

        Automatically determines the best threshold to separate
        foreground (dark marks) from background (white paper).

        Returns:
            Optimal threshold value (0-255)
        """
        if self.corrected is None:
            raise ValueError("No corrected image available.")

        # Convert to grayscale if needed
        if len(self.corrected.shape) == 3:
            gray = cv2.cvtColor(self.corrected, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.corrected

        # Use Otsu's method to find optimal threshold
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold * 0.50

        print(f"\n[Adaptive Threshold]")
        print(f"  50% Otsu's threshold: {threshold:.0f}")

        return int(threshold)

    def normalize_image(self, threshold: Optional[int] = None, use_binary: bool = True) -> np.ndarray:
        """
        Normalize image using adaptive thresholding or specified threshold.

        Creates a pure binary image (black=0, white=255) that separates
        filled marks from background/grid, reducing shadow effects.

        Args:
            threshold: Optional manual threshold (default: use Otsu's method)
            use_binary: If True, creates pure binary image. If False, returns CLAHE equalized.

        Returns:
            Binary normalized image (0 or 255 only)
        """
        if self.corrected is None:
            raise ValueError("No corrected image available.")

        # Convert to grayscale if needed
        if len(self.corrected.shape) == 3:
            gray = cv2.cvtColor(self.corrected, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.corrected

        print(f"\n[Image Normalization]")

        # Apply adaptive histogram equalization first to reduce shadow effects
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        print(f"  Applied CLAHE (Contrast Limited Adaptive Histogram Equalization)")

        if not use_binary:
            return equalized

        # Determine threshold on the equalized image
        if threshold is None:
            # Re-compute threshold on equalized image for better separation
            threshold, _ = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"  Otsu's threshold on equalized image: {threshold:.0f}")

        print(f"  Using threshold: {threshold}")

        # Apply binary threshold to create pure black/white image
        # THRESH_BINARY: pixels > threshold become white (255), others become black (0)
        _, binary = cv2.threshold(equalized, threshold, 255, cv2.THRESH_BINARY)

        print(f"  Created pure binary image (black=0, white=255)")
        print(f"  Goal: grid lines = white, filled marks = black (inverted later for detection)")

        return binary
