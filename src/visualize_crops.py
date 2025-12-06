"""
Visualize cropped credit card regions from bounding boxes.
Shows original vs occluded crops side by side.
"""
import json
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_coco_annotations(ann_file):
    """Load COCO annotations."""
    with open(ann_file, 'r') as f:
        return json.load(f)


def crop_bbox(image, bbox, padding=10):
    """
    Crop bounding box from image with padding.
    
    Args:
        image: Input image
        bbox: [x, y, width, height]
        padding: Padding pixels around bbox
    """
    h, w = image.shape[:2]
    x, y, bbox_w, bbox_h = bbox
    
    # Add padding
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(w, int(x + bbox_w + padding))
    y2 = min(h, int(y + bbox_h + padding))
    
    crop = image[y1:y2, x1:x2]
    return crop


def visualize_crops(
    original_dataset: str,
    occluded_dataset: str,
    num_samples: int = 6,
    output_path: str = "outputs/crop_visualization.png"
):
    """
    Visualize cropped regions from original and occluded datasets.
    
    Args:
        original_dataset: Path to original dataset
        occluded_dataset: Path to occluded dataset
        num_samples: Number of samples to show
        output_path: Output path for visualization
    """
    original_path = Path(original_dataset)
    occluded_path = Path(occluded_dataset)
    
    # Load annotations
    orig_ann_file = original_path / "train" / "_annotations.coco.json"
    occ_ann_file = occluded_path / "train" / "_annotations.coco.json"
    
    if not orig_ann_file.exists():
        orig_ann_file = original_path / "_annotations.coco.json"
    if not occ_ann_file.exists():
        occ_ann_file = occluded_path / "_annotations.coco.json"
    
    orig_data = load_coco_annotations(orig_ann_file)
    occ_data = load_coco_annotations(occ_ann_file)
    
    # Build lookup dictionaries
    orig_images = {img['id']: img for img in orig_data['images']}
    orig_anns = {img_id: [] for img_id in orig_images.keys()}
    for ann in orig_data['annotations']:
        orig_anns[ann['image_id']].append(ann)
    
    occ_images = {img['id']: img for img in occ_data['images']}
    occ_anns = {img_id: [] for img_id in occ_images.keys()}
    for ann in occ_data['annotations']:
        occ_anns[ann['image_id']].append(ann)
    
    # Find matching images
    orig_img_files = {img['file_name']: img for img in orig_data['images']}
    matching_images = []
    
    for img_name, img_info in orig_img_files.items():
        # Find corresponding image in occluded dataset
        occ_img = next((img for img in occ_data['images'] if img['file_name'] == img_name), None)
        if occ_img and orig_anns[img_info['id']] and occ_anns[occ_img['id']]:
            matching_images.append((img_info, occ_img))
    
    # Random sample
    if len(matching_images) > num_samples:
        matching_images = random.sample(matching_images, num_samples)
    
    # Create visualization
    rows = (num_samples + 1) // 2
    fig = plt.figure(figsize=(16, 8 * rows))
    gs = GridSpec(rows, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, (orig_img_info, occ_img_info) in enumerate(matching_images):
        row = idx // 2
        col_in_pair = idx % 2
        
        # Load images
        orig_img_path = original_path / "train" / orig_img_info['file_name']
        if not orig_img_path.exists():
            orig_img_path = original_path / orig_img_info['file_name']
        
        occ_img_path = occluded_path / "train" / occ_img_info['file_name']
        if not occ_img_path.exists():
            occ_img_path = occluded_path / occ_img_info['file_name']
        
        orig_img = cv2.imread(str(orig_img_path))
        occ_img = cv2.imread(str(occ_img_path))
        
        if orig_img is None or occ_img is None:
            continue
        
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        occ_img_rgb = cv2.cvtColor(occ_img, cv2.COLOR_BGR2RGB)
        
        # Get first annotation (assuming one card per image)
        orig_ann = orig_anns[orig_img_info['id']][0]
        occ_ann = occ_anns[occ_img_info['id']][0]
        
        # Crop regions
        orig_crop = crop_bbox(orig_img_rgb, orig_ann['bbox'], padding=20)
        occ_crop = crop_bbox(occ_img_rgb, occ_ann['bbox'], padding=20)
        
        # Plot original crop
        ax1 = fig.add_subplot(gs[row, col_in_pair * 2])
        ax1.imshow(orig_crop)
        ax1.set_title(f"Original\n{orig_img_info['file_name'][:30]}...", fontsize=10)
        ax1.axis('off')
        
        # Plot occluded crop
        ax2 = fig.add_subplot(gs[row, col_in_pair * 2 + 1])
        ax2.imshow(occ_crop)
        ax2.set_title(f"Occluded\n{occ_img_info['file_name'][:30]}...", fontsize=10)
        ax2.axis('off')
    
    # Add title
    fig.suptitle('Cropped Credit Card Regions: Original vs Occluded', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Crop visualization saved to: {output_path}")
    print(f"   Showing {len(matching_images)} sample crops")


def main():
    parser = argparse.ArgumentParser(description="Visualize cropped credit card regions")
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original dataset"
    )
    parser.add_argument(
        "--occluded",
        type=str,
        required=True,
        help="Path to occluded dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of samples to show (default: 6)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/crop_visualization.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    try:
        visualize_crops(
            original_dataset=args.original,
            occluded_dataset=args.occluded,
            num_samples=args.samples,
            output_path=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

