"""Tests for the WaifuDiffusion interrogator."""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tagger.interrogator.waifudiffusioninterrogator import WaifuDiffusionInterrogator


class TestWaifuDiffusionInterrogator:
    """Test the WaifuDiffusion interrogator implementation."""

    def test_init(self):
        """Test interrogator initialization."""
        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            model_path="test_model.onnx",
            tags_path="test_tags.csv",
            repo_id="test/repo"
        )

        assert interrogator.name == "wd14-test"
        assert interrogator.model_path == "test_model.onnx"
        assert interrogator.tags_path == "test_tags.csv"
        assert "repo_id" in interrogator.kwargs
        assert interrogator.kwargs["repo_id"] == "test/repo"

    @patch('tagger.interrogator.waifudiffusioninterrogator.hf_hub_download')
    @patch('tagger.interrogator.waifudiffusioninterrogator.logger')
    def test_download_not_quiet(self, mock_logger, mock_hf_download):
        """Test downloading model files when not in quiet mode."""
        mock_hf_download.side_effect = ["/path/to/model.onnx", "/path/to/tags.csv"]

        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )
        interrogator.set_quiet(False)

        model_path, tags_path = interrogator.download()

        assert model_path == Path("/path/to/model.onnx")
        assert tags_path == Path("/path/to/tags.csv")
        mock_logger.info.assert_called_once_with(
            "Loading model file", model="wd14-test", repo_id="test/repo"
        )

    @patch('tagger.interrogator.waifudiffusioninterrogator.hf_hub_download')
    @patch('tagger.interrogator.waifudiffusioninterrogator.logger')
    def test_download_quiet(self, mock_logger, mock_hf_download):
        """Test downloading model files in quiet mode."""
        mock_hf_download.side_effect = ["/path/to/model.onnx", "/path/to/tags.csv"]

        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )
        interrogator.set_quiet(True)

        model_path, tags_path = interrogator.download()

        assert model_path == Path("/path/to/model.onnx")
        assert tags_path == Path("/path/to/tags.csv")
        mock_logger.info.assert_not_called()

    @patch('tagger.interrogator.waifudiffusioninterrogator.pd.read_csv')
    @patch('tagger.interrogator.waifudiffusioninterrogator.logger')
    def test_load(self, mock_logger, mock_read_csv):
        """Test loading the model and tags."""
        # Setup mocks
        mock_tags_df = pd.DataFrame({"name": ["tag1", "tag2", "tag3"]})
        mock_read_csv.return_value = mock_tags_df
        mock_session = Mock()

        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )

        with patch.object(interrogator, 'download') as mock_download:
            with patch('onnxruntime.InferenceSession', return_value=mock_session):
                mock_download.return_value = (Path("/model.onnx"), Path("/tags.csv"))
                interrogator.load()

        assert interrogator.model == mock_session
        assert isinstance(interrogator.tags, pd.DataFrame)
        mock_logger.info.assert_called_once_with(
            "Model loaded", model="wd14-test", path="/model.onnx"
        )

    @patch('tagger.interrogator.waifudiffusioninterrogator.dbimutils')
    def test_interrogate_success(self, mock_dbimutils, sample_image, mock_onnx_session):
        """Test successful image interrogation."""
        # Setup interrogator
        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )
        interrogator.model = mock_onnx_session

        # Create mock tags DataFrame
        tags_data = {
            "name": ["general", "sensitive", "questionable", "explicit", "tag1", "tag2", "tag3"]
        }
        interrogator.tags = pd.DataFrame(tags_data)

        # Setup mock image processing
        mock_dbimutils.make_square.return_value = np.random.rand(224, 224, 3)
        mock_dbimutils.smart_resize.return_value = np.random.rand(224, 224, 3)

        # Setup mock model output
        mock_confidences = np.array([[0.1, 0.2, 0.3, 0.4, 0.8, 0.6, 0.9]])
        mock_onnx_session.run.return_value = [mock_confidences]
        mock_onnx_session.get_inputs.return_value = [Mock(shape=[1, 224, 3, 1], name="input")]
        mock_onnx_session.get_outputs.return_value = [Mock(name="output")]

        # Test interrogation
        ratings, tags = interrogator.interrogate(sample_image)

        # Verify results
        assert isinstance(ratings, dict)
        assert isinstance(tags, dict)
        assert len(ratings) == 4  # Should have 4 rating categories
        assert len(tags) == 3  # Should have 3 regular tags

    def test_interrogate_no_model_loads_automatically(self, sample_image):
        """Test that interrogate loads model if not already loaded."""
        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )

        with patch.object(interrogator, 'load') as mock_load:
            # Mock the load method to set up model and tags
            def setup_model():
                interrogator.model = Mock()
                interrogator.model.get_inputs.return_value = [Mock(shape=[1, 224, 3, 1], name="input")]
                interrogator.model.get_outputs.return_value = [Mock(name="output")]
                interrogator.model.run.return_value = [np.array([[0.1, 0.2, 0.3, 0.4, 0.8]])]
                interrogator.tags = pd.DataFrame({"name": ["general", "sensitive", "questionable", "explicit", "tag1"]})

            mock_load.side_effect = setup_model

            with patch('tagger.interrogator.waifudiffusioninterrogator.dbimutils'):
                ratings, tags = interrogator.interrogate(sample_image)

            mock_load.assert_called_once()

    def test_interrogate_no_model_raises_exception(self, sample_image):
        """Test that interrogate raises exception if model fails to load."""
        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )

        with patch.object(interrogator, 'load'):
            # Model remains None after load
            interrogator.model = None

            with pytest.raises(Exception, match="Model not loading"):
                interrogator.interrogate(sample_image)

    def test_interrogate_no_tags_raises_exception(self, sample_image, mock_onnx_session):
        """Test that interrogate raises exception if tags fail to load."""
        interrogator = WaifuDiffusionInterrogator(
            name="wd14-test",
            repo_id="test/repo"
        )
        interrogator.model = mock_onnx_session
        interrogator.tags = None

        with patch('tagger.interrogator.waifudiffusioninterrogator.dbimutils'):
            with pytest.raises(Exception, match="Tags not loading"):
                interrogator.interrogate(sample_image)