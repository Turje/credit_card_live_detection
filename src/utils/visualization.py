"""
Custom visualization utilities for object detection.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def get_class_colors(
    class_names: List[str], color_scheme: str = "default"
) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate colors for each class.

    Args:
        class_names: List of class names
        color_scheme: Color scheme ('default', 'bright', 'pastel')

    Returns:
        Dictionary mapping class names to BGR colors
    """
    colors = {}

    if color_scheme == "bright":
        color_palette = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
    elif color_scheme == "pastel":
        color_palette = [
            (144, 238, 144),  # Light Green
            (173, 216, 230),  # Light Blue
            (255, 182, 193),  # Light Pink
            (221, 160, 221),  # Plum
            (255, 218, 185),  # Peach
            (176, 224, 230),  # Powder Blue
            (255, 228, 196),  # Bisque
            (240, 255, 240),  # Honeydew
        ]
    else:  # default
        color_palette = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 191, 255),  # Deep Sky Blue
            (255, 20, 147),  # Deep Pink
        ]

    for i, class_name in enumerate(class_names):
        colors[class_name] = color_palette[i % len(color_palette)]

    return colors


def draw_custom_boxes(
    image: np.ndarray,
    detections: List[Dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    show_labels: bool = True,
    show_confidences: bool = True,
    label_format: str = "{class_name} {conf:.2f}",
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw custom bounding boxes on image.

    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with keys:
                   'x1', 'y1', 'x2', 'y2', 'class_name', 'confidence'
        class_colors: Optional dictionary mapping class names to BGR colors
        show_labels: Whether to show labels
        show_confidences: Whether to show confidence scores
        label_format: Format string for labels (use {class_name} and {conf})
        thickness: Box thickness
        font_scale: Font scale for labels

    Returns:
        Annotated image
    """
    annotated = image.copy()

    if not detections:
        return annotated

    # Get unique class names
    class_names = list(set([det.get("class_name", "unknown") for det in detections]))

    # Generate colors if not provided
    if class_colors is None:
        class_colors = get_class_colors(class_names)

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)

        # Get color for this class
        color = class_colors.get(class_name, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if show_labels:
            if show_confidences:
                label = label_format.format(class_name=class_name, conf=confidence)
            else:
                label = class_name

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 2),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )

    return annotated


def filter_detections_by_class(
    detections: List[Dict],
    class_names: Optional[List[str]] = None,
    exclude_classes: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Filter detections by class names.

    Args:
        detections: List of detection dictionaries
        class_names: List of class names to include (None = all)
        exclude_classes: List of class names to exclude

    Returns:
        Filtered list of detections
    """
    filtered = detections.copy()

    # Include only specified classes
    if class_names is not None:
        filtered = [det for det in filtered if det.get("class_name") in class_names]

    # Exclude specified classes
    if exclude_classes is not None:
        filtered = [
            det for det in filtered if det.get("class_name") not in exclude_classes
        ]

    return filtered


def apply_per_class_thresholds(
    detections: List[Dict],
    class_thresholds: Dict[str, float],
    default_threshold: float = 0.25,
) -> List[Dict]:
    """
    Apply per-class confidence thresholds.

    Args:
        detections: List of detection dictionaries
        class_thresholds: Dictionary mapping class names to thresholds
        default_threshold: Default threshold for classes not in dictionary

    Returns:
        Filtered list of detections
    """
    filtered = []

    for det in detections:
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)

        # Get threshold for this class
        threshold = class_thresholds.get(class_name, default_threshold)

        # Keep detection if confidence >= threshold
        if confidence >= threshold:
            filtered.append(det)

    return filtered
