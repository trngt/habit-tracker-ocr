"""
Configuration class for habit tracker scanner.

Centralizes all configurable parameters for grid dimensions,
detection thresholds, and image processing settings.
"""

from pathlib import Path


class TrackerConfig:
    """
    Configuration for the habit tracker scanner.

    Stores all parameters needed for:
    - Grid structure and dimensions
    - Corner marker detection
    - Bubble fill detection
    - Perspective correction output size

    Attributes:
        num_rows: Number of rows in the habit grid (days)
        num_cols: Number of columns in the habit grid (habits)
        grid_left: X offset from left edge of corrected image
        grid_top: Y offset from top edge of corrected image
        grid_width: Total grid width in pixels (corrected image)
        grid_height: Total grid height in pixels (corrected image)
        fill_threshold: Percentage threshold for detecting filled bubbles (0-1)
        cell_padding: Fraction of cell to ignore at borders (0-1)
        min_area: Minimum contour area for corner marker detection
        max_area: Maximum contour area for corner marker detection
        aspect_ratio_min: Minimum aspect ratio for square markers
        aspect_ratio_max: Maximum aspect ratio for square markers
        output_width: Width of perspective-corrected output image
        output_height: Height of perspective-corrected output image
        output_dir: Directory path for saving output images
    """

    def __init__(
        self,
        num_rows: int = 31,
        num_cols: int = 16,
        grid_left: int = 92,
        grid_top: int = 373,
        grid_width: int = 925,
        grid_height: int = 1737,
        fill_threshold: float = 0.75,
        cell_padding: float = 0.2,
        min_area: int = 500,
        max_area: int = 10000,
        aspect_ratio_min: float = 0.7,
        aspect_ratio_max: float = 1.3,
        output_width: int = 1700,
        output_height: int = 2200,
        output_dir: str = "./output"
    ):
        """
        Initialize configuration with default or custom parameters.

        Args:
            num_rows: Number of grid rows (default: 31 days)
            num_cols: Number of grid columns (default: 16 habits)
            grid_left: Grid left offset in pixels (default: 150)
            grid_top: Grid top offset in pixels (default: 400)
            grid_width: Grid width in pixels (default: 1000)
            grid_height: Grid height in pixels (default: 1650)
            fill_threshold: Fill detection threshold 0-1 (default: 0.15)
            cell_padding: Cell border padding fraction (default: 0.2)
            min_area: Min marker contour area (default: 500)
            max_area: Max marker contour area (default: 10000)
            aspect_ratio_min: Min square aspect ratio (default: 0.7)
            aspect_ratio_max: Max square aspect ratio (default: 1.3)
            output_width: Corrected image width (default: 1700)
            output_height: Corrected image height (default: 2200)
            output_dir: Output directory path (default: "./output")
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_left = grid_left
        self.grid_top = grid_top
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fill_threshold = fill_threshold
        self.cell_padding = cell_padding
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.output_width = output_width
        self.output_height = output_height
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """
        Validate that all configuration parameters are reasonable.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If any parameters are invalid
        """
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise ValueError("Grid dimensions must be positive")

        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("Grid size must be positive")

        if not 0 <= self.fill_threshold <= 1:
            raise ValueError("Fill threshold must be between 0 and 1")

        if not 0 <= self.cell_padding < 0.5:
            raise ValueError("Cell padding must be between 0 and 0.5")

        if self.min_area <= 0 or self.max_area <= self.min_area:
            raise ValueError("Invalid marker area constraints")

        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("Output dimensions must be positive")

        return True

    def to_dict(self) -> dict:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary of all configuration parameters
        """
        return {
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'grid_left': self.grid_left,
            'grid_top': self.grid_top,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'fill_threshold': self.fill_threshold,
            'cell_padding': self.cell_padding,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'aspect_ratio_min': self.aspect_ratio_min,
            'aspect_ratio_max': self.aspect_ratio_max,
            'output_width': self.output_width,
            'output_height': self.output_height,
            'output_dir': str(self.output_dir)
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrackerConfig':
        """
        Create TrackerConfig from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            TrackerConfig instance
        """
        return cls(**config_dict)
