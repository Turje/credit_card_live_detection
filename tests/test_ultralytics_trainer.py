"""
Unit tests for Ultralytics trainer.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from src.models.ultralytics_trainer import UltralyticsTrainer


class TestUltralyticsTrainer(unittest.TestCase):
    """Test cases for UltralyticsTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset"
        self.dataset_path.mkdir(parents=True)
        (self.dataset_path / "train").mkdir()
        (self.dataset_path / "val").mkdir()

    @patch("src.models.ultralytics_trainer.YOLO")
    def test_init_with_pretrained(self, mock_yolo):
        """Test initialization with pretrained model."""
        trainer = UltralyticsTrainer(model_name="yolov8n")

        mock_yolo.assert_called_once_with("yolov8n.pt")
        self.assertEqual(trainer.model_name, "yolov8n")
        self.assertIsNone(trainer.checkpoint)

    @patch("src.models.ultralytics_trainer.YOLO")
    def test_init_with_checkpoint(self, mock_yolo):
        """Test initialization with checkpoint."""
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pt"
        checkpoint_path.touch()

        trainer = UltralyticsTrainer(
            model_name="yolov8n", checkpoint=str(checkpoint_path)
        )

        mock_yolo.assert_called_once_with(str(checkpoint_path))
        self.assertEqual(trainer.checkpoint, str(checkpoint_path))

    @patch("src.models.ultralytics_trainer.YOLO")
    @patch("src.models.ultralytics_trainer.convert_coco_to_yolo")
    @patch("src.models.ultralytics_trainer.prepare_dataset_config")
    def test_train_basic(self, mock_prepare_config, mock_convert, mock_yolo):
        """Test basic training setup."""
        # Setup mocks
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = Path(self.temp_dir) / "results"
        (mock_results.save_dir / "weights").mkdir(parents=True)
        (mock_results.save_dir / "weights" / "best.pt").touch()
        (mock_results.save_dir / "weights" / "last.pt").touch()
        mock_model.train.return_value = mock_results
        mock_yolo.return_value = mock_model

        # Create dataset structure
        train_dir = self.dataset_path / "train"
        (train_dir / "labels").mkdir()

        trainer = UltralyticsTrainer(model_name="yolov8n")

        # Mock prepare_dataset_config to create a file
        mock_prepare_config.return_value = "dataset.yaml"

        results = trainer.train(
            dataset_path=self.dataset_path,
            epochs=10,
            img_size=640,
            batch_size=8,
            output_dir=Path(self.temp_dir) / "models",
            device="cpu",
        )

        # Verify model.train was called
        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        self.assertEqual(call_kwargs["epochs"], 10)
        self.assertEqual(call_kwargs["imgsz"], 640)
        self.assertEqual(call_kwargs["batch"], 8)
        self.assertEqual(call_kwargs["device"], "cpu")

    @patch("src.models.ultralytics_trainer.YOLO")
    def test_device_mapping(self, mock_yolo):
        """Test device name mapping."""
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_results.save_dir = Path(self.temp_dir) / "results"
        (mock_results.save_dir / "weights").mkdir(parents=True)
        (mock_results.save_dir / "weights" / "best.pt").touch()
        mock_model.train.return_value = mock_results
        mock_yolo.return_value = mock_model

        trainer = UltralyticsTrainer(model_name="yolov8n")

        # Test different device mappings
        for device_input, device_expected in [
            ("cuda", "0"),
            ("cpu", "cpu"),
            ("mps", "mps"),
        ]:
            with patch("src.models.ultralytics_trainer.convert_coco_to_yolo"), patch(
                "src.models.ultralytics_trainer.prepare_dataset_config",
                return_value="dataset.yaml",
            ):
                train_dir = self.dataset_path / "train"
                (train_dir / "labels").mkdir()

                trainer.train(
                    dataset_path=str(self.dataset_path), epochs=1, device=device_input
                )

                call_kwargs = mock_model.train.call_args[1]
                self.assertEqual(call_kwargs["device"], device_expected)


if __name__ == "__main__":
    unittest.main()
