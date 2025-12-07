"""
Unified inference script for video and image processing.
Currently supports: Ultralytics YOLOv8
Future: MMDetection (RTMDet, RT-DETR, YOLOX, DINO)
"""
import argparse
import sys
from pathlib import Path
from utils.config_loader import ConfigLoader
from models.ultralytics_inference import UltralyticsInference


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(
        description="Unified object detection inference"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file"
    )
    
    # Model (required)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Input (video or image)
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video (MP4)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (video or image)"
    )
    
    # Allow CLI overrides
    parser.add_argument("--framework", type=str, help="Model framework")
    parser.add_argument("--conf-threshold", type=float, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, help="IoU threshold")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu/mps)")
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.image:
        parser.error("Either --video or --image must be specified")
    
    if args.video and args.image:
        parser.error("Cannot specify both --video and --image")
    
    try:
        # Load configuration
        config_loader = ConfigLoader(config_path=args.config)
        config = config_loader.load(cli_args=args)
        
        # Override with CLI args if provided
        if args.video:
            config.inference.video_path = args.video
        if args.image:
            config.inference.image_path = args.image
        if args.output:
            config.inference.output_path = args.output
        if args.conf_threshold:
            config.inference.conf_threshold = args.conf_threshold
        if args.iou_threshold:
            config.inference.iou_threshold = args.iou_threshold
        if args.device:
            config.inference.device = args.device
        if args.no_save:
            config.inference.save_video = False
        if args.show:
            config.inference.show_video = True
        
        # Determine framework from CLI args, config, or model path
        if args.framework:
            framework = args.framework
        elif hasattr(config.model, 'framework') and config.model.framework:
            framework = config.model.framework
        else:
            # Infer from model file extension
            framework = "ultralytics"  # Default for .pt files
        
        # Infer framework from model path if not specified
        model_path = Path(args.model)
        if framework == "ultralytics" or model_path.suffix == ".pt":
            framework = "ultralytics"
        
        print("="*60)
        print("INFERENCE CONFIGURATION")
        print("="*60)
        print(f"Framework: {framework}")
        print(f"Model: {args.model}")
        if args.video:
            print(f"Input video: {args.video}")
        if args.image:
            print(f"Input image: {args.image}")
        print(f"Output: {config.inference.output_path}")
        print(f"Confidence threshold: {config.inference.conf_threshold}")
        print(f"IoU threshold: {config.inference.iou_threshold}")
        print(f"Device: {config.inference.device}")
        print("="*60)
        
        # Run inference based on framework
        if framework == "ultralytics":
            inference = UltralyticsInference(
                model_path=args.model,
                device=config.inference.device
            )
            
            if args.video:
                # Process video
                output_path = config.inference.output_path
                if not output_path.endswith('.mp4'):
                    output_path = str(Path(output_path) / Path(args.video).stem) + "_detected.mp4"
                
                inference.process_video(
                    video_path=args.video,
                    output_path=output_path,
                    conf_threshold=config.inference.conf_threshold,
                    iou_threshold=config.inference.iou_threshold,
                    save_video=config.inference.save_video,
                    show_video=config.inference.show_video
                )
            else:
                # Process image
                output_path = config.inference.output_path
                if Path(output_path).is_dir() or not output_path.endswith(('.jpg', '.png', '.jpeg')):
                    output_path = str(Path(output_path) / Path(args.image).name)
                
                inference.process_image(
                    image_path=args.image,
                    output_path=output_path,
                    conf_threshold=config.inference.conf_threshold,
                    iou_threshold=config.inference.iou_threshold
                )
        
        elif framework == "mmdet":
            print("❌ MMDetection support coming soon!")
            print("   Please use Ultralytics models (.pt files) for now")
            return 1
        else:
            print(f"❌ Unsupported framework: {framework}")
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

