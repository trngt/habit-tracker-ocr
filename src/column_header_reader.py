import cv2
import numpy as np
from typing import List, Optional


class ColumnHeaderReader:
    """
    Reads column header text from the habit tracker using OCR.

    Handles vertically-oriented (90-degree rotated) text above the grid.
    Uses EasyOCR for text recognition.

    Attributes:
        config: TrackerConfig instance with grid and header parameters
        reader: EasyOCR Reader instance (lazy-loaded)
        column_names: List of detected column names
    """

    def __init__(self, config):
        """
        Initialize ColumnHeaderReader with configuration.

        Args:
            config: TrackerConfig instance
        """
        self.config = config
        self._reader = None
        self.column_names: List[str] = []

    @property
    def reader(self):
        """Lazy-load EasyOCR reader to avoid slow startup if not needed."""
        if self._reader is None:
            import easyocr
            print("  Initializing EasyOCR reader...")
            self._reader = easyocr.Reader(['en'])
        return self._reader

    def extract_header_region(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the header region containing column names.

        Args:
            image: Perspective-corrected image (BGR or grayscale)

        Returns:
            Cropped image of the header region
        """
        x1 = self.config.grid_left
        x2 = x1 + self.config.grid_width
        y1 = self.config.header_top
        y2 = y1 + self.config.header_height

        return image[y1:y2, x1:x2].copy()

    def extract_column_slice(self, header_region: np.ndarray, col_index: int) -> np.ndarray:
        """
        Extract a single column's header slice.

        Args:
            header_region: Cropped header region image
            col_index: Column index (0-based)

        Returns:
            Cropped image of the single column header
        """
        col_width = header_region.shape[1] / self.config.num_cols
        x1 = int(col_index * col_width)
        x2 = int((col_index + 1) * col_width)

        return header_region[:, x1:x2].copy()

    def rotate_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image 90 degrees counter-clockwise for OCR.

        The column headers are written vertically (rotated 90 CW from normal),
        so we rotate them 90 CCW to make text horizontal for OCR.

        Args:
            image: Vertical text image

        Returns:
            Rotated image with horizontal text
        """
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def save_rotated_header_image(self, image: np.ndarray, output_path: str) -> np.ndarray:
        """
        Extract header region, rotate 90 degrees CCW, and save to file.

        Creates a visualization of the column headers with text
        oriented horizontally for easier inspection and OCR debugging.

        Args:
            image: Perspective-corrected image (BGR or grayscale)
            output_path: Where to save the rotated header image

        Returns:
            The rotated header region image
        """
        print(f"\n[Step 5.5] Extracting and rotating column header region...")
        print(f"  Header region: x={self.config.grid_left}-{self.config.grid_left + self.config.grid_width}, "
              f"y={self.config.header_top}-{self.config.header_top + self.config.header_height}")

        header_region = self.extract_header_region(image)
        print(f"  Extracted size: {header_region.shape[1]} x {header_region.shape[0]}")

        rotated = self.rotate_for_ocr(header_region)
        print(f"  Rotated size: {rotated.shape[1]} x {rotated.shape[0]}")

        cv2.imwrite(output_path, rotated)
        print(f"  Saved to {output_path}")

        return rotated

    def read_single_column(self, column_slice: np.ndarray) -> str:
        """
        Apply OCR to a single column header slice.

        Args:
            column_slice: Image of a single column header (vertical text)

        Returns:
            Detected text string, empty string if nothing detected
        """
        rotated = self.rotate_for_ocr(column_slice)
        results = self.reader.readtext(rotated)

        if results:
            # Combine all detected text fragments
            texts = [result[1] for result in results]
            return ' '.join(texts).strip()

        return ''

    def read_all_columns(self, image: np.ndarray) -> List[str]:
        """
        Read all column header names from the image.

        Args:
            image: Perspective-corrected image (BGR or grayscale)

        Returns:
            List of column names in order (length = num_cols)
        """
        print(f"\n[Step 6] Reading column headers with OCR...")
        print(f"  Header region: y={self.config.header_top} to {self.config.header_top + self.config.header_height}")
        print(f"  Columns to read: {self.config.num_cols}")

        header_region = self.extract_header_region(image)

        column_names = []
        for col in range(self.config.num_cols):
            col_slice = self.extract_column_slice(header_region, col)
            name = self.read_single_column(col_slice)
            column_names.append(name)

            if name:
                print(f"    Column {col + 1}: '{name}'")
            else:
                print(f"    Column {col + 1}: (no text detected)")

        self.column_names = column_names
        print(f"  Total columns with text: {sum(1 for n in column_names if n)}")

        return column_names

    def visualize_header_regions(
        self,
        output_path: str,
        image: np.ndarray,
        column_names: Optional[List[str]] = None
    ) -> None:
        """
        Create visualization showing header regions and detected text.

        Draws bounding boxes around each column header region and
        annotates with the detected text.

        Args:
            output_path: Where to save visualization image
            image: Original image to annotate
            column_names: List of column names (uses self.column_names if None)
        """
        if column_names is None:
            column_names = self.column_names

        print(f"\n[Visualization] Creating column header annotation...")

        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Draw header region boundary
        x1 = self.config.grid_left
        x2 = x1 + self.config.grid_width
        y1 = self.config.header_top
        y2 = y1 + self.config.header_height

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta

        # Draw each column slice and annotate
        col_width = self.config.grid_width / self.config.num_cols

        for col in range(self.config.num_cols):
            col_x1 = int(x1 + col * col_width)
            col_x2 = int(x1 + (col + 1) * col_width)

            # Draw column boundary
            cv2.rectangle(vis, (col_x1, y1), (col_x2, y2), (255, 200, 0), 1)  # Cyan

            # Add detected text annotation
            if col < len(column_names) and column_names[col]:
                text = column_names[col]
                text_x = col_x1 + int(col_width / 2) - 5
                char_height = 20
                for i, char in enumerate(text[:15]):  # Limit to 15 chars
                    text_y = y2 - 10 - (len(text[:15]) - 1 - i) * char_height
                    cv2.putText(vis, char, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add title
        cv2.putText(vis, "Column Header Detection", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(vis, f"Detected: {sum(1 for n in column_names if n)}/{len(column_names)} columns",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(output_path, vis)
        print(f"  Column header visualization saved to {output_path}")

    def create_header_debug_image(self, output_path: str, image: np.ndarray) -> None:
        """
        Create debug image showing extracted and rotated column slices.

        Useful for debugging OCR issues.

        Args:
            output_path: Where to save debug image
            image: Perspective-corrected image
        """
        print(f"\n[Debug] Creating header extraction debug image...")

        header_region = self.extract_header_region(image)

        slices = []
        max_width = 0

        for col in range(self.config.num_cols):
            col_slice = self.extract_column_slice(header_region, col)
            rotated = self.rotate_for_ocr(col_slice)

            if len(rotated.shape) == 2:
                rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

            slices.append(rotated)
            max_width = max(max_width, rotated.shape[1])

        # Pad slices to same width and stack vertically
        padded_slices = []
        for i, s in enumerate(slices):
            pad_width = max_width - s.shape[1]
            if pad_width > 0:
                padding = np.ones((s.shape[0], pad_width, 3), dtype=np.uint8) * 255
                s = np.hstack([s, padding])

            # Add column number label
            cv2.putText(s, f"Col {i+1}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            padded_slices.append(s)

        # Stack all slices vertically with separators
        separator = np.ones((5, max_width, 3), dtype=np.uint8) * 128

        result_parts = []
        for s in padded_slices:
            result_parts.append(s)
            result_parts.append(separator)

        result = np.vstack(result_parts[:-1])  # Remove last separator

        cv2.imwrite(output_path, result)
        print(f"  Header debug image saved to {output_path}")
