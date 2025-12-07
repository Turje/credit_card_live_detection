"""
Ultralytics YOLOv8 inference wrapper for video processing.
"""
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Tuple
import cv2
import numpy as np


class UltralyticsInference:
    """Wrapper for Ultralytics YOLOv8 inference."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize Ultralytics inference.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device (cuda, cpu, mps)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Map device names
        device_map = {
            "cuda": "0",
            "cpu": "cpu",
            "mps": "mps"
        }
        self.device = device_map.get(device, device)
        
        print(f"Loading model: {model_path}")
        self.model = YOLO(str(self.model_path))
        print(f"✅ Model loaded successfully")
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_video: bool = True,
        show_video: bool = False
    ) -> Tuple[int, int]:
        """
        Process video file and save results.
        
        Args:
            video_path: Path to input video (MP4)
            output_path: Path to output video
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_video: Whether to save output video
            show_video: Whether to display video during processing
            
        Returns:
            Tuple of (total_frames, processed_frames)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
        
        frame_count = 0
        processed_frames = 0
        
        print(f"\n🔄 Processing video...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Draw bounding boxes
                annotated_frame = results[0].plot()
                
                # Save frame
                if save_video:
                    out.write(annotated_frame)
                
                # Show frame
                if show_video:
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                processed_frames += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
            if save_video:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        print(f"\n✅ Video processing complete!")
        print(f"   Processed {processed_frames}/{total_frames} frames")
        if save_video:
            print(f"   Output saved to: {output_path}")
        
        return total_frames, processed_frames
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> np.ndarray:
        """
        Process single image and return annotated result.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Annotated image as numpy array
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model.predict(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_image)
            print(f"✅ Image saved to: {output_path}")
        
        return annotated_image

