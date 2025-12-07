"""
Unified training script supporting multiple frameworks.
Currently supports: Ultralytics YOLOv8
Future: MMDetection (RTMDet, RT-DETR, YOLOX, DINO)
"""

import argparse
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from models.ultralytics_trainer import UltralyticsTrainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Unified object detection training")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )

    # Allow CLI overrides
    parser.add_argument("--framework", type=str, help="Model framework")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--dataset-path", type=str, help="Dataset path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--img-size", type=int, help="Image size")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu/mps)")

    args = parser.parse_args()

    try:
        # Load configuration
        config_loader = ConfigLoader(config_path=args.config)
        config = config_loader.load(cli_args=args)

        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Framework: {config.model.framework}")
        print(f"Model: {config.model.name}")
        print(f"Dataset: {config.dataset.path}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"Image size: {config.training.img_size}")
        print(f"Device: {config.training.device}")
        print("=" * 60)

        # Train based on framework
        if config.model.framework == "ultralytics":
            trainer = UltralyticsTrainer(
                model_name=config.model.name, checkpoint=config.model.checkpoint
            )
            trainer.train(
                dataset_path=config.dataset.path,
                epochs=config.training.epochs,
                img_size=config.training.img_size,
                batch_size=config.training.batch_size,
                output_dir=config.training.output_dir,
                device=config.training.device,
                workers=config.training.workers,
                patience=config.training.patience,
            )
        elif config.model.framework == "mmdet":
            print("❌ MMDetection support coming soon!")
            print("   Please use --framework ultralytics for now")
            return 1
        else:
            print(f"❌ Unsupported framework: {config.model.framework}")
            print("   Supported: ultralytics, mmdet")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
