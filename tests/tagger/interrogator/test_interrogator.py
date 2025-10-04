"""Tests for the base interrogator class."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tagger.interrogator.interrogator import AbsInterrogator


class TestAbsInterrogator:
    """Test the abstract interrogator base class."""

    def test_init(self):
        """Test interrogator initialization."""
        interrogator = AbsInterrogator("test-model")

        assert interrogator.name == "test-model"
        assert interrogator.quiet is False
        assert interrogator.providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def test_set_quiet(self):
        """Test setting quiet mode."""
        interrogator = AbsInterrogator("test-model")

        interrogator.set_quiet(True)
        assert interrogator.quiet is True

        interrogator.set_quiet(False)
        assert interrogator.quiet is False

    @patch('tagger.interrogator.interrogator.logger')
    def test_unload_with_model(self, mock_logger):
        """Test unloading when model is loaded."""
        interrogator = AbsInterrogator("test-model")
        interrogator.model = Mock()
        interrogator.tags = {"tag1": 0.5}

        result = interrogator.unload()

        assert result is True
        assert not hasattr(interrogator, 'model') or interrogator.model is None
        assert not hasattr(interrogator, 'tags')
        mock_logger.info.assert_called_once_with("Model unloaded", model="test-model")

    @patch('tagger.interrogator.interrogator.logger')
    def test_unload_without_model(self, mock_logger):
        """Test unloading when no model is loaded."""
        interrogator = AbsInterrogator("test-model")

        result = interrogator.unload()

        assert result is False
        mock_logger.info.assert_not_called()

    @patch('tagger.interrogator.interrogator.logger')
    def test_unload_quiet_mode(self, mock_logger):
        """Test unloading in quiet mode doesn't log."""
        interrogator = AbsInterrogator("test-model")
        interrogator.model = Mock()
        interrogator.set_quiet(True)

        result = interrogator.unload()

        assert result is True
        mock_logger.info.assert_not_called()

    @patch('tagger.interrogator.interrogator.logger')
    def test_use_cpu(self, mock_logger):
        """Test forcing CPU execution."""
        interrogator = AbsInterrogator("test-model")

        interrogator.use_cpu()

        assert interrogator.providers == ['CPUExecutionProvider']
        mock_logger.info.assert_called_once_with("Forcing CPU execution", model="test-model")

    @patch('tagger.interrogator.interrogator.get_available_providers')
    def test_get_available_providers(self, mock_get_providers):
        """Test getting available providers."""
        mock_get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        interrogator = AbsInterrogator("test-model")

        result = interrogator.get_available_providers()

        assert result == ['CUDAExecutionProvider', 'CPUExecutionProvider']
        mock_get_providers.assert_called_once()

    def test_set_providers(self):
        """Test setting custom providers."""
        interrogator = AbsInterrogator("test-model")
        custom_providers = ['CPUExecutionProvider', 'CustomProvider']

        interrogator.set_providers(custom_providers)

        assert interrogator.providers == custom_providers

    def test_postprocess_tags_basic(self):
        """Test basic tag postprocessing."""
        tags = {
            "high_conf_tag": 0.8,
            "medium_conf_tag": 0.5,
            "low_conf_tag": 0.2
        }

        result = AbsInterrogator.postprocess_tags(
            tags,
            threshold=0.4,
            escape_tag=False,
            replace_underscore=False
        )

        expected = {
            "high_conf_tag": 0.8,
            "medium_conf_tag": 0.5
        }
        assert result == expected

    def test_postprocess_tags_with_exclusions(self):
        """Test tag postprocessing with exclusions."""
        tags = {
            "good_tag": 0.8,
            "excluded_tag": 0.7,
            "another_good_tag": 0.6
        }

        result = AbsInterrogator.postprocess_tags(
            tags,
            threshold=0.5,
            exclude_tags={"excluded_tag"}
        )

        expected = {
            "good_tag": 0.8,
            "another_good_tag": 0.6
        }
        assert result == expected

    def test_postprocess_tags_with_additional_tags(self):
        """Test tag postprocessing with additional tags."""
        tags = {
            "existing_tag": 0.8
        }

        result = AbsInterrogator.postprocess_tags(
            tags,
            threshold=0.5,
            additional_tags=["additional_tag1", "additional_tag2"]
        )

        expected = {
            "existing_tag": 0.8,
            "additional_tag1": 1.0,
            "additional_tag2": 1.0
        }
        assert result == expected

    def test_postprocess_tags_escape_and_replace(self):
        """Test tag postprocessing with escaping and underscore replacement."""
        tags = {
            "tag_with_underscore": 0.8,
            "tag(with)parentheses": 0.7
        }

        result = AbsInterrogator.postprocess_tags(
            tags,
            threshold=0.5,
            escape_tag=True,
            replace_underscore=True
        )

        expected = {
            "tag with underscore": 0.8,
            "tag\\(with\\)parentheses": 0.7
        }
        assert result == expected

    def test_postprocess_tags_no_escape_no_replace(self):
        """Test tag postprocessing without escaping or underscore replacement."""
        tags = {
            "tag_with_underscore": 0.8,
            "tag(with)parentheses": 0.7
        }

        result = AbsInterrogator.postprocess_tags(
            tags,
            threshold=0.5,
            escape_tag=False,
            replace_underscore=False
        )

        expected = {
            "tag_with_underscore": 0.8,
            "tag(with)parentheses": 0.7
        }
        assert result == expected

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        interrogator = AbsInterrogator("test-model")

        with pytest.raises(NotImplementedError):
            interrogator.load()

        with pytest.raises(NotImplementedError):
            interrogator.interrogate(Mock())