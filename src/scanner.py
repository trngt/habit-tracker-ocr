import pandas as pd
from typing import Optional
from pathlib import Path

from .config import TrackerConfig
from .image_processor import ImageProcessor
from .grid_bubble_detector import GridBubbleDetector


class HabitTrackerScanner:
    """
    Main class that orchestrates the habit tracker scanning process.
    
    Coordinates ImageProcessor and GridBubbleDetector to:
    1. Load and correct scanned images
    2. Detect filled bubbles in the grid
    3. Return results as a pandas DataFrame
    
    Attributes:
        config: TrackerConfig instance with grid parameters
        image_processor: ImageProcessor instance for image operations
        grid_bubble_detector: GridBubbleDetector instance for detection
    """
    
    def __init__(self, config: TrackerConfig):
        """
        Initialize the scanner with configuration.

        Args:
            config: TrackerConfig instance with grid and detection parameters
        """
        self.config = config
        self.config.validate()
        self.image_processor = ImageProcessor(config)
        self.grid_bubble_detector = GridBubbleDetector(config)
        self.results = None
        
    def scan(self, image_path: str) -> pd.DataFrame:
        """
        Main pipeline: processes image and detects filled bubbles.

        Args:
            image_path: Path to the scanned habit tracker image

        Returns:
            pandas DataFrame with detection results
            Columns: ['day', 'habit_column', 'is_filled']
            or similar structure TBD

        Raises:
            ValueError: If image cannot be loaded or processed
            Exception: For other processing errors (gracefully handled)
        """
        # Validate image path
        validated_path = self._validate_image_path(image_path)

        print("=" * 60)
        print("HABIT TRACKER SCANNER")
        print("=" * 60)

        # Step 1-3: Image processing
        print("\n[Step 1] Loading and preprocessing image...")
        self.image_processor.load_image(str(validated_path))
        self.image_processor.preprocess()

        print("\n[Step 2] Detecting ArUco corner markers...")
        self.image_processor.detect_aruco_markers()

        # Visualize detection
        output_path = self.config.output_dir / "detected_markers.jpg"
        self.image_processor.visualize_corners(str(output_path))

        print("\n[Step 3] Applying perspective correction...")
        corrected = self.image_processor.apply_perspective_correction()

        # Save corrected image
        output_path = self.config.output_dir / "corrected_grid.jpg"
        import cv2
        cv2.imwrite(str(output_path), corrected)
        print(f"\nCorrected image saved to {output_path}")

        # Also save grayscale version
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(self.config.output_dir / "corrected_grid_gray.jpg"), corrected_gray)
        print(f"Grayscale version saved to corrected_grid_gray.jpg")

        # Step 3.5: Histogram analysis and image normalization
        print("\n[Step 3.5] Analyzing histogram and normalizing image...")

        # Determine threshold (use config value or compute automatically)
        if self.config.normalization_threshold is not None:
            threshold = self.config.normalization_threshold
            print(f"Using manual threshold from config: {threshold}")
        else:
            threshold = self.image_processor.compute_adaptive_threshold()

        # Visualize histogram with threshold marked
        output_path = self.config.output_dir / "histogram.png"
        self.image_processor.visualize_histogram(str(output_path), threshold)

        # Normalize image to pure binary (black/white only)
        normalized = self.image_processor.normalize_image(threshold, use_binary=True)

        # Save normalized image
        output_path = self.config.output_dir / "normalized_image.jpg"
        cv2.imwrite(str(output_path), normalized)
        print(f"Normalized image saved to {output_path}")

        # Step 4: Grid calculation
        self.grid_bubble_detector.set_image(normalized)
        self.grid_bubble_detector.calculate_grid_cells()

        # Visualize grid positioning
        output_path = self.config.output_dir / "grid_annotation.jpg"
        self.grid_bubble_detector.visualize_grid(str(output_path), corrected)

        # Step 5: Bubble detection
        detection_results = self.grid_bubble_detector.detect_all_bubbles()

        # Create visualizations
        output_path = self.config.output_dir / "bubble_detection.jpg"
        self.grid_bubble_detector.visualize_detection(str(output_path), corrected)

        output_path = self.config.output_dir / "fill_ratios_heatmap.jpg"
        self.grid_bubble_detector.create_fill_ratio_heatmap(str(output_path))

        # Print summary
        self._print_results_summary(detection_results)

        # Store results
        self.results = detection_results

        print("\n" + "=" * 60)
        print("SUCCESS! Bubble detection complete.")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - detected_markers.jpg      (corner marker detection)")
        print("  - corrected_grid.jpg        (perspective-corrected image)")
        print("  - histogram.png             (pixel intensity distribution)")
        print("  - normalized_image.jpg      (contrast-enhanced image)")
        print("  - grid_annotation.jpg       (grid cell positioning)")
        print("  - bubble_detection.jpg      (annotated with detected fills)")
        print("  - fill_ratios_heatmap.jpg   (heatmap showing fill ratios)")

        return self.results
    
    def get_results(self) -> pd.DataFrame:
        """
        Retrieve the most recent detection results.

        Returns:
            pandas DataFrame with detection results from last scan
        """
        return self.results

    def _validate_image_path(self, image_path: str) -> Path:
        """
        Validate that image path exists and is readable.

        Args:
            image_path: Path to image file

        Returns:
            Path object if valid

        Raises:
            ValueError: If path is invalid or file doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        return path

    def _print_results_summary(self, detection_results: dict) -> None:
        """
        Print human-readable summary of detection results.

        Args:
            detection_results: Dictionary of day -> filled habit columns
        """
        print("\n" + "=" * 60)
        print("DETECTION RESULTS SUMMARY")
        print("=" * 60)

        if not detection_results:
            print("No filled bubbles detected.")
            return

        print(f"\nTotal days with activity: {len(detection_results)}")
        print(f"Total filled bubbles: {sum(len(cols) for cols in detection_results.values())}")

        print("\nDetailed results:")
        for day in sorted(detection_results.keys()):
            habit_cols = detection_results[day]
            # Convert to 1-indexed for display
            habit_numbers = [col + 1 for col in habit_cols]
            print(f"  Day {day:2d}: Habits {habit_numbers}")

        print("=" * 60)


def main():
    """
    Main function to run the habit tracker scanner.
    """
    # Configuration
    config = TrackerConfig(
        output_dir="./output"
    )

    # Initialize scanner
    scanner = HabitTrackerScanner(config)

    # Scan the sample image
    image_path = "./input/version_2_filled.jpg"
    scanner.scan(image_path)


if __name__ == "__main__":
    main()
