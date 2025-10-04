"""Tests for the CamieTagger interrogator."""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
from pathlib import Path
import numpy as np
import json
from PIL import Image

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tagger.interrogator.camietaggerinterrogator import CamieTaggerInterrogator, preprocess_image


class TestCamieTaggerInterrogator:
    """Test the CamieTagger interrogator implementation."""

    def test_init(self):
        """Test interrogator initialization."""
        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx",
            tags_path="metadata.json"
        )

        assert interrogator.name == "camie-test"
        assert interrogator.repo_id == "test/repo"
        assert interrogator.model_path == "model.onnx"
        assert interrogator.tags_path == "metadata.json"
        assert interrogator.tags is None
        assert interrogator.model is None

    @patch('tagger.interrogator.camietaggerinterrogator.hf_hub_download')
    @patch('tagger.interrogator.camietaggerinterrogator.logger')
    def test_download_success(self, mock_logger, mock_hf_download):
        """Test successful model download."""
        mock_hf_download.side_effect = ["/path/to/model.onnx", "/path/to/metadata.json"]

        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx"
        )

        model_path, tags_path = interrogator.download()

        assert model_path == "/path/to/model.onnx"
        assert tags_path == "/path/to/metadata.json"
        mock_logger.info.assert_called_once_with(
            "Loading model file", model="camie-test", repo_id="test/repo"
        )

    @patch('tagger.interrogator.camietaggerinterrogator.InferenceSession')
    @patch('builtins.open', mock_open(read_data='{"idx_to_tag": {"0": "tag1"}, "tag_to_category": {"tag1": "general"}}'))
    @patch('tagger.interrogator.camietaggerinterrogator.logger')
    def test_load_success(self, mock_logger, mock_inference_session):
        """Test successful model loading."""
        mock_session = Mock()
        mock_inference_session.return_value = mock_session

        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx"
        )

        with patch.object(interrogator, 'download') as mock_download:
            mock_download.return_value = ("/model.onnx", "/metadata.json")
            interrogator.load()

        assert interrogator.model == mock_session
        assert interrogator.metadata == {
            "idx_to_tag": {"0": "tag1"},
            "tag_to_category": {"tag1": "general"}
        }
        mock_logger.info.assert_called_once_with(
            "Model loaded", model="camie-test", path="/model.onnx"
        )

    def test_interrogate_loads_model_if_needed(self, sample_image):
        """Test that interrogate loads model if not already loaded."""
        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx"
        )

        with patch.object(interrogator, 'load') as mock_load:
            with patch.object(interrogator, 'model', None):
                # Mock the load method to set up model
                def setup_model():
                    interrogator.model = Mock()
                    interrogator.model.get_inputs.return_value = [Mock(name="input", type="tensor(float)")]
                    interrogator.model.run.return_value = [np.array([[0.1, 0.8]])]
                    interrogator.metadata = {
                        "idx_to_tag": {"0": "rating_tag", "1": "general_tag"},
                        "tag_to_category": {"rating_tag": "rating", "general_tag": "general"}
                    }

                mock_load.side_effect = setup_model

                with patch('tagger.interrogator.camietaggerinterrogator.preprocess_image') as mock_preprocess:
                    mock_preprocess.return_value = np.random.rand(3, 512, 512)
                    ratings, tags = interrogator.interrogate(sample_image)

                mock_load.assert_called_once()

    def test_interrogate_raises_exception_if_no_model(self, sample_image):
        """Test that interrogate raises exception if model is None after load."""
        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx"
        )

        with patch.object(interrogator, 'load'):
            interrogator.model = None

            with pytest.raises(Exception, match="Model not loading"):
                interrogator.interrogate(sample_image)

    @patch('tagger.interrogator.camietaggerinterrogator.preprocess_image')
    def test_interrogate_success(self, mock_preprocess, sample_image):
        """Test successful image interrogation."""
        # Setup interrogator
        interrogator = CamieTaggerInterrogator(
            name="camie-test",
            repo_id="test/repo",
            model_path="model.onnx"
        )

        # Setup mock model
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_input.type = "tensor(float)"
        mock_model.get_inputs.return_value = [mock_input]

        # Mock model outputs - two outputs for initial and refined predictions
        initial_output = np.array([[0.1, 0.8, 0.6, 0.3]])  # Logits
        refined_output = np.array([[0.2, 0.9, 0.7, 0.4]])  # Refined logits
        mock_model.run.return_value = [initial_output, refined_output]

        interrogator.model = mock_model

        # Setup metadata
        interrogator.metadata = {
            "idx_to_tag": {
                "0": "rating_safe",
                "1": "character_tag",
                "2": "general_tag1",
                "3": "general_tag2"
            },
            "tag_to_category": {
                "rating_safe": "rating",
                "character_tag": "character",
                "general_tag1": "general",
                "general_tag2": "general"
            }
        }

        # Setup preprocessing mock
        mock_preprocess.return_value = np.random.rand(3, 512, 512).astype(np.float32)

        # Test interrogation
        ratings, tags = interrogator.interrogate(sample_image)

        # Verify results
        assert isinstance(ratings, dict)
        assert isinstance(tags, dict)
        mock_preprocess.assert_called_once_with(sample_image)


class TestPreprocessImage:
    """Test the preprocess_image function."""

    def test_preprocess_rgb_image(self, sample_image):
        """Test preprocessing RGB image."""
        result = preprocess_image(sample_image, image_size=256)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 256, 256)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_preprocess_rgba_image(self, sample_rgba_image):
        """Test preprocessing RGBA image (should convert to RGB)."""
        result = preprocess_image(sample_rgba_image, image_size=256)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 256, 256)
        assert result.dtype == np.float32

    def test_preprocess_different_aspect_ratios(self):
        """Test preprocessing images with different aspect ratios."""
        # Create wide image
        wide_image = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        result_wide = preprocess_image(wide_image, image_size=256)

        # Create tall image
        tall_image = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
        result_tall = preprocess_image(tall_image, image_size=256)

        # Both should result in 256x256 output
        assert result_wide.shape == (3, 256, 256)
        assert result_tall.shape == (3, 256, 256)

    def test_preprocess_custom_size(self, sample_image):
        """Test preprocessing with custom image size."""
        result = preprocess_image(sample_image, image_size=128)

        assert result.shape == (3, 128, 128)
        assert result.dtype == np.float32