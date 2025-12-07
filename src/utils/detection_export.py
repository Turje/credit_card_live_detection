"""
Utilities for exporting detection results to various formats.
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def export_detections_json(
    detections: List[Dict[str, Any]],
    output_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export detections to JSON format.

    Args:
        detections: List of detection dictionaries
        output_path: Path to output JSON file
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "detections": detections,
    }

    if metadata:
        export_data["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ JSON export saved to: {output_path}")


def export_detections_csv(
    detections: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Export detections to CSV format.

    Args:
        detections: List of detection dictionaries
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not detections:
        print("⚠️  No detections to export")
        return

    fieldnames = list(detections[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)

    print(f"✅ CSV export saved to: {output_path}")


def export_detections_xml(
    detections: List[Dict[str, Any]],
    output_path: str | Path,
    image_path: Optional[str] = None,
    image_size: Optional[tuple] = None,
) -> None:
    """
    Export detections to Pascal VOC XML format.

    Args:
        detections: List of detection dictionaries
        output_path: Path to output XML file
        image_path: Path to source image
        image_size: Image size (width, height)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("annotation")

    # Add folder and filename
    if image_path:
        folder_elem = ET.SubElement(root, "folder")
        folder_elem.text = str(Path(image_path).parent.name)
        filename_elem = ET.SubElement(root, "filename")
        filename_elem.text = Path(image_path).name

    # Add source
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Object Detection"

    # Add image size
    if image_size:
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_size[0])
        height = ET.SubElement(size, "height")
        height.text = str(image_size[1])
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

    # Add segmented
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    # Add objects
    for det in detections:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = str(det.get("class", "unknown"))
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bbox, "xmin")
        xmin.text = str(int(det.get("x1", 0)))
        ymin = ET.SubElement(bbox, "ymin")
        ymin.text = str(int(det.get("y1", 0)))
        xmax = ET.SubElement(bbox, "xmax")
        xmax.text = str(int(det.get("x2", 0)))
        ymax = ET.SubElement(bbox, "ymax")
        ymax.text = str(int(det.get("y2", 0)))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ XML export saved to: {output_path}")


def parse_ultralytics_results(
    results, frame_number: Optional[int] = None, image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Parse Ultralytics results into standardized detection format.

    Args:
        results: Ultralytics results object
        frame_number: Optional frame number for video
        image_path: Optional image path

    Returns:
        List of detection dictionaries
    """
    detections: List[Dict[str, Any]] = []

    if not results or len(results) == 0:
        return detections

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for i, box in enumerate(boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detection = {
            "detection_id": i,
            "class_id": int(box.cls[0].cpu().numpy()),
            "class_name": result.names[int(box.cls[0].cpu().numpy())],
            "confidence": float(box.conf[0].cpu().numpy()),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        }

        if frame_number is not None:
            detection["frame_number"] = frame_number

        if image_path:
            detection["image_path"] = str(image_path)

        detections.append(detection)

    return detections
