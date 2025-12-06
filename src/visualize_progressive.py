"""
Visualize progressive occlusion on a single image.
Shows original, 25%, 50%, 75% occlusion side by side.
"""
import json
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_coco_annotations(ann_file):
    """Load COCO annotations."""
    with open(ann_file, 'r') as f:
        return json.load(f)


def visualize_progressive_occlusion(
    original_dataset: str,
    occlusion_datasets: dict,  # {25: path, 50: path, 75: path}
    output_path: str = "outputs/progressive_occlusion_comparison.png",
    image_name: str = None
):
    """
    Visualize progressive occlusion on a single image.
    
    Args:
        original_dataset: Path to original dataset
        occlusion_datasets: Dict mapping occlusion levels to dataset paths
        output_path: Output path for visualization
        image_name: Specific image to visualize (None = random)
    """
    original_path = Path(original_dataset)
    
    # Load original annotations
    orig_ann_file = original_path / "train" / "_annotations.coco.json"
    if not orig_ann_file.exists():
        orig_ann_file = original_path / "_annotations.coco.json"
    
    orig_data = load_coco_annotations(orig_ann_file)
    
    # Select image
    if image_name:
        orig_img_info = next((img for img in orig_data['images'] if img['file_name'] == image_name), None)
        if not orig_img_info:
            print(f"Image {image_name} not found in original dataset")
            return
    else:
        # Random image
        orig_img_info = random.choice(orig_data['images'])
    
    img_filename = orig_img_info['file_name']
    print(f"Visualizing image: {img_filename}")
    
    # Load original image
    orig_img_path = original_path / "train" / img_filename
    if not orig_img_path.exists():
        orig_img_path = original_path / img_filename
    
    orig_img = cv2.imread(str(orig_img_path))
    if orig_img is None:
        print(f"Could not load original image: {orig_img_path}")
        return
    
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Load occluded images
    occluded_images = {}
    for level, dataset_path in occlusion_datasets.items():
        occ_path = Path(dataset_path)
        occ_img_path = occ_path / "train" / img_filename
        if not occ_img_path.exists():
            occ_img_path = occ_path / img_filename
        
        if occ_img_path.exists():
            occ_img = cv2.imread(str(occ_img_path))
            if occ_img is not None:
                occluded_images[level] = cv2.cvtColor(occ_img, cv2.COLOR_BGR2RGB)
            else:
                print(f"Warning: Could not load {level}% occluded image")
        else:
            print(f"Warning: {level}% occluded image not found: {occ_img_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(orig_img_rgb)
    axes[0].set_title('Original (0%)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Occluded versions
    for idx, level in enumerate([25, 50, 75], 1):
        if level in occluded_images:
            axes[idx].imshow(occluded_images[level])
            axes[idx].set_title(f'Occluded ({level}%)', fontsize=14, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, f'{level}% occlusion\nnot available', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'Occluded ({level}%)', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle(f'Progressive Occlusion: {img_filename}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Progressive occlusion visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize progressive occlusion on single image")
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original dataset"
    )
    parser.add_argument(
        "--occlusion-25",
        type=str,
        required=True,
        help="Path to 25% occlusion dataset"
    )
    parser.add_argument(
        "--occlusion-50",
        type=str,
        required=True,
        help="Path to 50% occlusion dataset"
    )
    parser.add_argument(
        "--occlusion-75",
        type=str,
        required=True,
        help="Path to 75% occlusion dataset"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image filename to visualize (default: random)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/progressive_occlusion_comparison.png",
        help="Output path for visualization"
    )
    
    args = parser.parse_args()
    
    occlusion_datasets = {
        25: args.occlusion_25,
        50: args.occlusion_50,
        75: args.occlusion_75
    }
    
    try:
        visualize_progressive_occlusion(
            original_dataset=args.original,
            occlusion_datasets=occlusion_datasets,
            output_path=args.output,
            image_name=args.image
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

