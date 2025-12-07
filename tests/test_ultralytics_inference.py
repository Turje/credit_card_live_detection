"""
Unit tests for Ultralytics inference.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
from src.models.ultralytics_inference import UltralyticsInference


class TestUltralyticsInference(unittest.TestCase):
    """Test cases for UltralyticsInference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model.pt"
        self.model_path.touch()
    
    @patch('src.models.ultralytics_inference.YOLO')
    def test_init(self, mock_yolo):
        """Test initialization."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        inference = UltralyticsInference(
            model_path=str(self.model_path),
            device="cpu"
        )
        
        mock_yolo.assert_called_once_with(str(self.model_path))
        self.assertEqual(inference.device, "cpu")
    
    @patch('src.models.ultralytics_inference.YOLO')
    def test_init_nonexistent_model(self, mock_yolo):
        """Test error when model doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            UltralyticsInference(model_path="nonexistent.pt")
    
    @patch('src.models.ultralytics_inference.YOLO')
    @patch('src.models.ultralytics_inference.cv2.VideoCapture')
    @patch('src.models.ultralytics_inference.cv2.VideoWriter')
    def test_process_video(self, mock_video_writer, mock_video_capture, mock_yolo):
        """Test video processing."""
        import cv2
        
        # Setup mocks
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FRAME_COUNT: 10
        }.get(x, 0)
        
        # Mock frame reading
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame)] * 10 + [(False, None)]
        
        mock_video_capture.return_value = mock_cap
        
        # Mock video writer
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        inference = UltralyticsInference(
            model_path=str(self.model_path),
            device="cpu"
        )
        
        video_path = Path(self.temp_dir) / "test.mp4"
        video_path.touch()
        output_path = Path(self.temp_dir) / "output.mp4"
        
        total_frames, processed_frames = inference.process_video(
            video_path=str(video_path),
            output_path=str(output_path),
            conf_threshold=0.25,
            save_video=True
        )
        
        self.assertEqual(processed_frames, 10)
        self.assertEqual(total_frames, 10)
        mock_model.predict.assert_called()
        mock_writer.write.assert_called()
    
    @patch('src.models.ultralytics_inference.YOLO')
    def test_process_image(self, mock_yolo):
        """Test image processing."""
        # Setup mocks
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        inference = UltralyticsInference(
            model_path=str(self.model_path),
            device="cpu"
        )
        
        image_path = Path(self.temp_dir) / "test.jpg"
        image_path.touch()
        output_path = Path(self.temp_dir) / "output.jpg"
        
        result = inference.process_image(
            image_path=str(image_path),
            output_path=str(output_path),
            conf_threshold=0.25
        )
        
        self.assertIsInstance(result, np.ndarray)
        mock_model.predict.assert_called_once()
    
    @patch('src.models.ultralytics_inference.YOLO')
    def test_device_mapping(self, mock_yolo):
        """Test device name mapping."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Test different device mappings
        device_tests = [
            ("cuda", "0"),
            ("cpu", "cpu"),
            ("mps", "mps")
        ]
        
        for device_input, device_expected in device_tests:
            inference = UltralyticsInference(
                model_path=str(self.model_path),
                device=device_input
            )
            self.assertEqual(inference.device, device_expected)


if __name__ == '__main__':
    unittest.main()

