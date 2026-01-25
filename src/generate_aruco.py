#!/usr/bin/env python3
"""
Generate ArUco markers for habit tracker grid calibration
Creates 4 markers (100x100 pixels) with IDs 0-3 for the four grid corners
"""

import cv2
import numpy as np
from pathlib import Path


def generate_aruco_markers(output_dir: str = "aruco_markers", marker_size: int = 100):
    """
    Generate 4 ArUco markers (100x100) for grid corners.
    
    Marker IDs:
        0 = Top-Left (TL)
        1 = Top-Right (TR)
        2 = Bottom-Left (BL)
        3 = Bottom-Right (BR)
    
    Args:
        output_dir: Directory to save marker PNGs
        marker_size: Size of each marker in pixels (default 100x100)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load ArUco dictionary (4x4 with 50 unique markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Marker labels
    labels = {
        0: "TL",  # Top-Left
        1: "TR",  # Top-Right
        2: "BL",  # Bottom-Left
        3: "BR"   # Bottom-Right
    }
    
    print(f"Generating ArUco markers ({marker_size}x{marker_size} pixels)...")
    print(f"Dictionary: DICT_4X4_50")
    print()
    
    for marker_id, label in labels.items():
        # Generate marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Save marker
        filename = f"marker_{label}.png"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), marker_image)
        
        print(f"✓ Generated: {filename} (ID: {marker_id})")
    
    print(f"\n✓ All markers saved to: {output_dir}/")
    print("\nUsage:")
    print("  - Place marker_TL.png at top-left corner of your grid")
    print("  - Place marker_TR.png at top-right corner of your grid")
    print("  - Place marker_BL.png at bottom-left corner of your grid")
    print("  - Place marker_BR.png at bottom-right corner of your grid")


if __name__ == "__main__":
    generate_aruco_markers()