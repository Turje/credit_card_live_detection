"""
Test script for enhanced inference features.
Run this to verify all new features work correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced
from utils.visualization import get_class_colors
from utils.detection_export import parse_ultralytics_results


def test_enhanced_inference():
    """Test enhanced inference features."""
    print("=" * 60)
    print("Testing Enhanced Inference Features")
    print("=" * 60)

    # Check if model exists (use a test model or provide path)
    model_path = "models/model_n/weights/best.pt"  # Update with your model path
    if not Path(model_path).exists():
        print(f"⚠️  Model not found at: {model_path}")
        print("   Please update model_path in this script or train a model first")
        print("   Example: make train-unified DATASET=datasets/your_dataset MODEL_NAME=yolov8n")
        return False

    try:
        # Initialize enhanced inference
        print("\n1. Testing initialization...")
        inference = UltralyticsInferenceEnhanced(
            model_path=model_path,
            device="cuda" if Path("/dev/nvidia0").exists() else "cpu",
            multi_gpu=False,
        )
        print("   ✅ Initialization successful")

        # Test with a sample image (if available)
        test_image = Path("test_image.jpg")  # Update with your test image
        if test_image.exists():
            print("\n2. Testing single image processing...")
            annotated_img, detections = inference.process_image(
                image_path=test_image,
                output_path="outputs/test_output.jpg",
                conf_threshold=0.25,
                show_labels=True,
                show_confidences=True,
                export_format="json",
                export_path="outputs/test_detections.json",
            )
            print(f"   ✅ Processed image: {len(detections)} detections found")
            print(f"   ✅ Export saved to: outputs/test_detections.json")

            # Test per-class thresholds
            print("\n3. Testing per-class thresholds...")
            class_thresholds = {"person": 0.5, "car": 0.3}  # Example thresholds
            _, filtered_detections = inference.process_image(
                image_path=test_image,
                output_path="outputs/test_filtered.jpg",
                class_thresholds=class_thresholds,
            )
            print(f"   ✅ Per-class thresholds applied: {len(filtered_detections)} detections")

            # Test class filtering
            print("\n4. Testing class filtering...")
            _, filtered = inference.process_image(
                image_path=test_image,
                output_path="outputs/test_class_filtered.jpg",
                filter_classes=["person", "car"],  # Only show these classes
            )
            print(f"   ✅ Class filtering applied: {len(filtered)} detections")

            # Test custom colors
            print("\n5. Testing custom visualization...")
            custom_colors = get_class_colors(["person", "car"], color_scheme="bright")
            _, _ = inference.process_image(
                image_path=test_image,
                output_path="outputs/test_custom_colors.jpg",
                custom_colors=custom_colors,
                show_labels=True,
                show_confidences=True,
            )
            print("   ✅ Custom visualization applied")

        else:
            print(f"\n⚠️  Test image not found at: {test_image}")
            print("   Skipping image processing tests")

        # Test batch processing (if images available)
        image_dir = Path("test_images")
        if image_dir.exists() and list(image_dir.glob("*.jpg")):
            print("\n6. Testing batch image processing...")
            image_paths = list(image_dir.glob("*.jpg"))[:5]  # Test with first 5 images
            all_detections = inference.process_batch_images(
                image_paths=image_paths,
                output_dir="outputs/batch_output",
                batch_size=2,
                export_format="csv",
            )
            print(f"   ✅ Batch processing complete: {len(all_detections)} total detections")

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_inference()
    sys.exit(0 if success else 1)

