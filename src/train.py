"""
YOLOv8 Training Script for Credit Card Detection
Supports training on full and partial occlusion datasets
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def prepare_dataset_config(dataset_path: str, output_config: str = "dataset.yaml"):
    """
    Create YOLOv8 dataset config file.
    
    Args:
        dataset_path: Path to dataset directory
        output_config: Output config file path
    """
    dataset_path = Path(dataset_path)
    
    # Find annotation file
    ann_file = dataset_path / "train" / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    # Load COCO to get class names
    import json
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data.get('categories', [])
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    
    # Check for split dataset structure
    dataset_parent = dataset_path.parent
    split_base = dataset_parent / f"{dataset_path.name.replace('_split', '')}_split"
    
    # Determine paths
    train_path = 'train'
    val_path = 'val' if (split_base / "val").exists() else 'train'
    
    # Create YOLOv8 config
    config = {
        'path': str(split_base.absolute()) if split_base.exists() else str(dataset_path.absolute()),
        'train': f'{train_path}',
        'val': f'{val_path}',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    # Write config
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config created: {output_config}")
    print(f"Classes: {class_names}")
    
    return output_config


def convert_coco_to_yolo(dataset_path: str):
    """
    Convert COCO format to YOLO format.
    
    Args:
        dataset_path: Path to COCO dataset
    """
    import json
    import shutil
    from pathlib import Path
    
    dataset_path = Path(dataset_path)
    
    # Determine if dataset_path is the base directory or already the train directory
    # Check if annotation file exists directly (dataset_path is train directory)
    ann_file = dataset_path / "_annotations.coco.json"
    if ann_file.exists():
        train_dir = dataset_path
        base_dir = dataset_path.parent
    else:
        # Check if it's in train subdirectory (dataset_path is base directory)
        train_dir = dataset_path / "train"
        ann_file = train_dir / "_annotations.coco.json"
        base_dir = dataset_path
    
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found. Checked:\n"
            f"  - {dataset_path / '_annotations.coco.json'}\n"
            f"  - {dataset_path / 'train' / '_annotations.coco.json'}"
        )
    
    # Load COCO annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Create YOLO directory structure
    yolo_train = train_dir / "images"
    yolo_labels = train_dir / "labels"
    yolo_train.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create label files
    for img_id, img_info in images.items():
        img_filename = img_info['file_name']
        img_path = train_dir / img_filename
        
        if not img_path.exists():
            continue
        
        # Copy image
        shutil.copy2(img_path, yolo_train / img_filename)
        
        # Get annotations for this image
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # Create YOLO label file
        label_file = yolo_labels / (Path(img_filename).stem + '.txt')
        img_w = img_info['width']
        img_h = img_info['height']
        
        with open(label_file, 'w') as f:
            for ann in img_anns:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                center_x = (x + w / 2) / img_w
                center_y = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                category_id = ann['category_id']
                # Find category index (0-based)
                cat_ids = sorted(categories.keys())
                class_id = cat_ids.index(category_id)
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    print(f"Converted COCO to YOLO format")
    print(f"Images: {len(list(yolo_train.glob('*')))}")
    print(f"Labels: {len(list(yolo_labels.glob('*.txt')))}")


def train_model(
    dataset_path: str,
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    output_dir: str = "models"
):
    """
    Train YOLOv8 model.
    
    Args:
        dataset_path: Path to dataset directory
        model_size: Model size (nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        output_dir: Output directory for models
    """
    # Check if split dataset exists
    dataset_parent = Path(dataset_path).parent
    split_base = dataset_parent / f"{Path(dataset_path).name.replace('_split', '')}_split"
    
    if split_base.exists():
        train_path = split_base / "train"
        val_path = split_base / "val"
    else:
        train_path = Path(dataset_path) / "train"
        val_path = Path(dataset_path) / "val" if (Path(dataset_path) / "val").exists() else train_path
    
    # Convert COCO to YOLO format if needed
    yolo_labels = train_path / "labels"
    if not yolo_labels.exists() or len(list(yolo_labels.glob("*.txt"))) == 0:
        print("Converting COCO to YOLO format...")
        # Convert train set
        convert_coco_to_yolo(str(train_path.parent) if split_base.exists() else dataset_path)
        # Convert val set if separate
        if val_path != train_path and val_path.exists():
            convert_coco_to_yolo(str(val_path.parent) if split_base.exists() else str(val_path.parent))
    
    # Create dataset config
    config_file = "dataset.yaml"
    prepare_dataset_config(dataset_path, config_file)
    
    # Initialize model
    model = YOLO(f"yolov8{model_size}.pt")  # Load pretrained
    
    # Train
    results = model.train(
        data=config_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name=f"credit_card_{model_size}",
        exist_ok=True,
        save=True,
        plots=True
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for credit card detection")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n(nano), s(small), m(medium), l(large), x(xlarge)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            dataset_path=args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            output_dir=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

