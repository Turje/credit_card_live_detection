"""
Unit tests for config loader.
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import argparse
from src.utils.config_loader import ConfigLoader, Config


class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

        # Create test config
        test_config = {
            "model": {
                "framework": "ultralytics",
                "name": "yolov8n",
                "checkpoint": None,
            },
            "dataset": {
                "path": "datasets/test_dataset",
                "format": "coco",
                "train_split": "train",
                "val_split": "val",
            },
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "img_size": 640,
                "output_dir": "models",
                "device": "cuda",
                "workers": 8,
                "patience": 50,
            },
            "inference": {
                "video_path": None,
                "image_path": None,
                "output_path": "outputs",
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "device": "cuda",
                "save_video": True,
                "show_video": False,
            },
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        loader = ConfigLoader(config_path=str(self.config_path))
        config_dict = loader.load_from_yaml()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model"]["framework"], "ultralytics")
        self.assertEqual(config_dict["model"]["name"], "yolov8n")
        self.assertEqual(config_dict["dataset"]["path"], "datasets/test_dataset")

    def test_load_from_nonexistent_file(self):
        """Test error handling for nonexistent config file."""
        loader = ConfigLoader(config_path="nonexistent.yaml")

        with self.assertRaises(FileNotFoundError):
            loader.load_from_yaml()

    def test_merge_configs_with_cli(self):
        """Test merging YAML config with CLI arguments."""
        loader = ConfigLoader(config_path=str(self.config_path))
        yaml_config = loader.load_from_yaml()

        # Create mock CLI args
        cli_args = argparse.Namespace(
            framework="mmdet",
            model_name="rtmdet-s",
            dataset_path="datasets/custom",
            epochs=50,
            batch_size=32,
            device="cpu",
        )

        config = loader.merge_configs(yaml_config, cli_args)

        # CLI args should override YAML
        self.assertEqual(config.model.framework, "mmdet")
        self.assertEqual(config.model.name, "rtmdet-s")
        self.assertEqual(config.dataset.path, "datasets/custom")
        self.assertEqual(config.training.epochs, 50)
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.training.device, "cpu")

        # Non-overridden values should remain from YAML
        self.assertEqual(config.training.img_size, 640)
        self.assertEqual(config.inference.conf_threshold, 0.25)

    def test_load_complete(self):
        """Test complete load process."""
        loader = ConfigLoader(config_path=str(self.config_path))
        config = loader.load()

        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.framework, "ultralytics")
        self.assertEqual(config.model.name, "yolov8n")
        self.assertEqual(config.dataset.path, "datasets/test_dataset")
        self.assertEqual(config.training.epochs, 100)

    def test_get_config_before_load(self):
        """Test error when getting config before loading."""
        loader = ConfigLoader(config_path=str(self.config_path))

        with self.assertRaises(ValueError):
            loader.get_config()

    def test_get_config_after_load(self):
        """Test getting config after loading."""
        loader = ConfigLoader(config_path=str(self.config_path))
        config1 = loader.load()
        config2 = loader.get_config()

        self.assertEqual(config1, config2)


if __name__ == "__main__":
    unittest.main()
