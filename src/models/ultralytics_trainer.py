"""
Ultralytics YOLOv8 training wrapper.
"""
from pathlib import Path
from ultralytics import YOLO
import yaml
from typing import Optional
from ..train import convert_coco_to_yolo, prepare_dataset_config


class UltralyticsTrainer:
    """Wrapper for Ultralytics YOLOv8 training."""
    
    def __init__(self, model_name: str = "yolov8n", checkpoint: Optional[str] = None):
        """
        Initialize Ultralytics trainer.
        
        Args:
            model_name: Model name (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            checkpoint: Path to checkpoint file (None = use pretrained)
        """
        self.model_name = model_name
        self.checkpoint = checkpoint
        
        # Load model
        if checkpoint and Path(checkpoint).exists():
            print(f"Loading checkpoint: {checkpoint}")
            self.model = YOLO(checkpoint)
        else:
            model_file = f"{model_name}.pt"
            print(f"Loading pretrained model: {model_file}")
            self.model = YOLO(model_file)
    
    def train(
        self,
        dataset_path: str,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        output_dir: str = "models",
        device: str = "cuda",
        workers: int = 8,
        patience: int = 50
    ):
        """
        Train YOLOv8 model.
        
        Args:
            dataset_path: Path to dataset directory
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size
            output_dir: Output directory for models
            device: Device (cuda, cpu, mps)
            workers: Number of data loading workers
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        dataset_path = Path(dataset_path)
        
        # Check if split dataset exists
        dataset_parent = dataset_path.parent
        split_base = dataset_parent / f"{dataset_path.name.replace('_split', '')}_split"
        
        if split_base.exists():
            train_path = split_base / "train"
            val_path = split_base / "val"
        else:
            train_path = dataset_path / "train"
            val_path = dataset_path / "val" if (dataset_path / "val").exists() else train_path
        
        # Convert COCO to YOLO format if needed
        yolo_labels = train_path / "labels"
        if not yolo_labels.exists() or len(list(yolo_labels.glob("*.txt"))) == 0:
            print("Converting COCO to YOLO format...")
            # Convert train set
            convert_coco_to_yolo(str(train_path.parent) if split_base.exists() else str(dataset_path))
            # Convert val set if separate
            if val_path != train_path and val_path.exists():
                convert_coco_to_yolo(str(val_path.parent) if split_base.exists() else str(val_path.parent))
        
        # Create dataset config
        config_file = "dataset.yaml"
        prepare_dataset_config(str(dataset_path), config_file)
        
        # Map device names
        device_map = {
            "cuda": "0",
            "cpu": "cpu",
            "mps": "mps"
        }
        device_str = device_map.get(device, device)
        
        # Train
        results = self.model.train(
            data=config_file,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=output_dir,
            name=f"{self.model_name.replace('yolov8', 'model')}",
            exist_ok=True,
            save=True,
            plots=True,
            device=device_str,
            workers=workers,
            patience=patience
        )
        
        print(f"\n✅ Training complete!")
        print(f"📁 Best model: {results.save_dir}/weights/best.pt")
        print(f"📁 Latest model: {results.save_dir}/weights/last.pt")
        
        return results

