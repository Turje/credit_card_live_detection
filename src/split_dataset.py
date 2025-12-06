"""
Split COCO dataset into train/val/test sets.
"""
import json
import argparse
import random
from pathlib import Path
import shutil
from collections import defaultdict


def split_dataset(
    dataset_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Split COCO dataset into train/val/test sets.
    
    Args:
        dataset_path: Path to dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    ann_file = train_dir / "_annotations.coco.json"
    
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    # Load COCO annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Set random seed
    random.seed(random_seed)
    
    # Get all images
    images = coco_data['images']
    random.shuffle(images)
    
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count  # Remaining goes to test
    
    print(f"Splitting {total_images} images:")
    print(f"  Train: {train_count} ({train_count/total_images*100:.1f}%)")
    print(f"  Val: {val_count} ({val_count/total_images*100:.1f}%)")
    print(f"  Test: {test_count} ({test_count/total_images*100:.1f}%)")
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count+val_count]
    test_images = images[train_count+val_count:]
    
    # Create output directories
    output_base = dataset_path.parent / f"{dataset_path.name}_split"
    output_base.mkdir(exist_ok=True)
    
    for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        split_dir = output_base / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Get image IDs for this split
        split_image_ids = {img['id'] for img in split_images}
        
        # Filter annotations for this split
        split_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] in split_image_ids
        ]
        
        # Copy images
        for img_info in split_images:
            src_img = train_dir / img_info['file_name']
            if src_img.exists():
                shutil.copy2(src_img, split_dir / img_info['file_name'])
        
        # Create new COCO annotation file
        split_coco_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': split_images,
            'annotations': split_annotations
        }
        
        # Update image IDs to be sequential
        img_id_mapping = {old_id: new_id for new_id, old_id in enumerate([img['id'] for img in split_images], 1)}
        
        for img in split_coco_data['images']:
            img['id'] = img_id_mapping[img['id']]
        
        for ann in split_coco_data['annotations']:
            ann['image_id'] = img_id_mapping[ann['image_id']]
            ann['id'] = len(split_coco_data['annotations'])  # Simple ID assignment
        
        # Reassign annotation IDs sequentially
        for i, ann in enumerate(split_coco_data['annotations'], 1):
            ann['id'] = i
        
        # Save annotation file
        ann_output = split_dir / "_annotations.coco.json"
        with open(ann_output, 'w') as f:
            json.dump(split_coco_data, f, indent=2)
        
        print(f"\n{split_name.upper()} set created:")
        print(f"  Images: {len(split_images)}")
        print(f"  Annotations: {len(split_annotations)}")
        print(f"  Location: {split_dir}")
    
    print(f"\n✅ Dataset split complete!")
    print(f"Output directory: {output_base}")
    
    return output_base


def main():
    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val/test")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            dataset_path=args.dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

