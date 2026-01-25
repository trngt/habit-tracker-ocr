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

        print("\n[Step 2] Detecting corner markers...")
        self.image_processor.detect_corner_markers()

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

        # Step 4: Grid calculation
        self.grid_bubble_detector.set_image(corrected)
        self.grid_bubble_detector.calculate_grid_cells()

        # Visualize grid positioning
        output_path = self.config.output_dir / "grid_annotation.jpg"
        self.grid_bubble_detector.visualize_grid(str(output_path), corrected)

        # TODO: Step 5 (bubble detection) will be implemented next
        print("\n" + "=" * 60)
        print("SUCCESS! Grid calculation complete.")
        print("=" * 60)

        return None  # Will return results later
    
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
    image_path = "./input/IMG_1370.jpg"
    scanner.scan(image_path)


if __name__ == "__main__":
    main()
