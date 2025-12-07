# Quick Test Guide for Enhanced Inference Features

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Have a trained model:**
   - Train a model first: `make train-unified DATASET=datasets/your_dataset MODEL_NAME=yolov8n`
   - Or use a pretrained YOLOv8 model (will download automatically)

3. **Have test images/videos:**
   - Place test images in `test_images/` directory
   - Or use a single test image/video

## Quick Test Methods

### Method 1: Using Python Script (Recommended)

```bash
# Update model_path and test_image in test_enhanced_inference.py
python test_enhanced_inference.py
```

### Method 2: Using Python Interactively

```python
from src.models.ultralytics_inference_enhanced import UltralyticsInferenceEnhanced

# Initialize
inference = UltralyticsInferenceEnhanced(
    model_path="models/model_n/weights/best.pt",  # Your model path
    device="cuda"  # or "cpu" or "mps"
)

# Process single image
annotated_img, detections = inference.process_image(
    image_path="test_image.jpg",
    output_path="outputs/result.jpg",
    conf_threshold=0.25,
    show_labels=True,
    show_confidences=True,
    export_format="json",
    export_path="outputs/detections.json"
)

print(f"Found {len(detections)} detections")
```

### Method 3: Test Individual Features

#### Test Per-Class Thresholds
```python
inference = UltralyticsInferenceEnhanced(model_path="your_model.pt")

# Different thresholds for different classes
class_thresholds = {
    "person": 0.5,  # Higher threshold for person
    "car": 0.3,     # Lower threshold for car
}

annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    class_thresholds=class_thresholds
)
```

#### Test Class Filtering
```python
# Only show specific classes
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    filter_classes=["person", "car"]  # Only these classes
)

# Exclude specific classes
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    exclude_classes=["background", "noise"]  # Exclude these
)
```

#### Test Custom Visualization
```python
from src.utils.visualization import get_class_colors

# Get custom colors
custom_colors = get_class_colors(["person", "car"], color_scheme="bright")

annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    custom_colors=custom_colors,
    show_labels=True,
    show_confidences=True
)
```

#### Test Detection Export
```python
# Export to JSON
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="json",
    export_path="outputs/detections.json"
)

# Export to CSV
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="csv",
    export_path="outputs/detections.csv"
)

# Export to XML (Pascal VOC format)
annotated_img, detections = inference.process_image(
    image_path="test.jpg",
    export_format="xml",
    export_path="outputs/detections.xml"
)
```

#### Test Batch Processing
```python
from pathlib import Path

# Process multiple images
image_paths = list(Path("test_images").glob("*.jpg"))

all_detections = inference.process_batch_images(
    image_paths=image_paths,
    output_dir="outputs/batch_results",
    batch_size=8,
    export_format="json"
)
```

#### Test Video Processing with Frame Skipping
```python
# Process video, skipping every other frame (2x faster)
total_frames, processed_frames, detections = inference.process_video(
    video_path="test_video.mp4",
    output_path="outputs/result.mp4",
    frame_skip=2,  # Process every 2nd frame
    export_format="json",
    export_path="outputs/video_detections.json"
)
```

#### Test Multi-GPU (if available)
```python
# Use multiple GPUs
inference = UltralyticsInferenceEnhanced(
    model_path="your_model.pt",
    device="cuda",
    multi_gpu=True,
    gpu_ids=[0, 1]  # Use GPU 0 and 1
)
```

## Expected Outputs

After running tests, you should see:

1. **Processed images** in `outputs/` directory
2. **Export files** (JSON/CSV/XML) with detection data
3. **Console output** showing:
   - Number of detections found
   - Processing progress
   - Export file locations

## Troubleshooting

1. **Model not found:**
   - Train a model first or use pretrained YOLOv8 model
   - Update model path in test script

2. **CUDA errors:**
   - Use `device="cpu"` if no GPU available
   - Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Import errors:**
   - Make sure you're in the project root directory
   - Run: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

4. **No detections:**
   - Lower confidence threshold: `conf_threshold=0.1`
   - Check if your model supports the classes in your test images

## Next Steps

Once verified, integrate into your workflow:
1. Update `src/inference.py` to use `UltralyticsInferenceEnhanced`
2. Update `src/utils/config_loader.py` to handle new config fields
3. Add CLI arguments for new features

