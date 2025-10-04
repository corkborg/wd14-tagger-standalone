"""Tests for the main run.py script."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import os

# Add the project root to the path so we can import tagger modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tagger import cli as run


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_parse_exclude_tags_none(self):
        """Test parsing exclude tags when none provided."""
        mock_args = Mock()
        mock_args.exclude_tags = None
        result = run.parse_exclude_tags(mock_args)
        assert result == set()

    def test_parse_exclude_tags_single(self):
        """Test parsing single exclude tag."""
        mock_args = Mock()
        mock_args.exclude_tags = ["tag1,tag2,tag_with_underscore"]
        result = run.parse_exclude_tags(mock_args)
        expected = {
            "tag1", "tag2", "tag_with_underscore",
            "tag1", "tag2", "tag_with_underscore"  # reverse escaped versions
        }
        assert result == expected

    def test_parse_additional_tags_none(self):
        """Test parsing additional tags when none provided."""
        mock_args = Mock()
        mock_args.additional_tags = None
        result = run.parse_additional_tags(mock_args)
        assert result == []

    def test_parse_additional_tags_with_duplicates(self):
        """Test parsing additional tags removes duplicates."""
        mock_args = Mock()
        mock_args.additional_tags = ["tag1,tag2", "tag2,tag3"]
        result = run.parse_additional_tags(mock_args)
        assert set(result) == {"tag1", "tag2", "tag3"}


class TestImageProcessing:
    """Test image processing functions."""

    @patch('tagger.cli.AbsInterrogator.postprocess_tags')
    def test_image_interrogate_success(self, mock_postprocess, sample_image):
        """Test successful image interrogation."""
        # Setup mocks
        mock_interrogator = Mock()
        mock_interrogator.interrogate.return_value = ({}, {"tag1": 0.8, "tag2": 0.6})
        mock_postprocess.return_value = {"tag1": 0.8}

        mock_args = Mock()
        mock_args.threshold = 0.35

        # Test
        with patch('tagger.cli.Image.open', return_value=sample_image):
            result = run.image_interrogate(
                Path("test.jpg"),
                tag_escape=True,
                exclude_tags=set(),
                additional_tags=[],
                args=mock_args,
                interrogator=mock_interrogator
            )

        # Verify
        assert isinstance(result, dict)
        mock_interrogator.interrogate.assert_called_once()
        mock_postprocess.assert_called_once()

    def test_explore_image_files_no_recursive(self, tmp_path):
        """Test exploring image files without recursion."""
        # Create test files
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()
        (tmp_path / "not_image.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "image3.webp").touch()

        # Test
        result = list(run.explore_image_files(tmp_path, recursive=False))

        # Should only find files in root, not subdirectory
        assert len(result) == 2
        filenames = [f.name for f in result]
        assert "image1.jpg" in filenames
        assert "image2.png" in filenames
        assert "image3.webp" not in filenames

    def test_explore_image_files_recursive(self, tmp_path):
        """Test exploring image files with recursion."""
        # Create test files
        (tmp_path / "image1.jpg").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "image2.webp").touch()

        # Test
        result = list(run.explore_image_files(tmp_path, recursive=True))

        # Should find files in both root and subdirectory
        assert len(result) == 2
        filenames = [f.name for f in result]
        assert "image1.jpg" in filenames
        assert "image2.webp" in filenames


class TestOutputGeneration:
    """Test output string generation."""

    def test_generate_output_string_text_format(self):
        """Test generating text format output."""
        mock_args = Mock()
        mock_args.json = False

        tags = {"tag1": 0.8, "tag2": 0.6, "tag3": 0.4}
        result = run.generate_output_string("test.jpg", tags, mock_args)

        assert result == "tag1, tag2, tag3"

    def test_generate_output_string_json_format(self):
        """Test generating JSON format output."""
        import json

        mock_args = Mock()
        mock_args.json = True

        tags = {"tag1": 0.8, "tag2": 0.6}
        result = run.generate_output_string("test.jpg", tags, mock_args)

        # Parse the JSON to verify structure
        parsed = json.loads(result)
        assert parsed["file"] == "test.jpg"
        assert parsed["tags"] == tags


class TestLoggingSetup:
    """Test logging configuration."""

    @patch('tagger.cli.structlog')
    @patch('tagger.cli.logging')
    def test_setup_logging_without_progress_bar(self, mock_logging, mock_structlog):
        """Test logging setup without progress bar."""
        mock_logger = Mock()
        mock_structlog.get_logger.return_value = mock_logger

        result = run.setup_logging(enable_progress_bar=False)

        # Verify structlog was configured
        mock_structlog.configure.assert_called_once()
        assert result == mock_logger

    @patch('tagger.cli.structlog')
    @patch('tagger.cli.logging')
    def test_setup_logging_with_progress_bar(self, mock_logging, mock_structlog):
        """Test logging setup with progress bar."""
        mock_logger = Mock()
        mock_structlog.get_logger.return_value = mock_logger

        result = run.setup_logging(enable_progress_bar=True)

        # Verify structlog was configured
        mock_structlog.configure.assert_called_once()
        assert result == mock_logger


class TestTqdmWriter:
    """Test the TqdmWriter class."""

    def test_tqdm_writer_no_progress_bar(self):
        """Test TqdmWriter when no progress bar is active."""
        with patch('tagger.cli._current_progress_bar', None):
            with patch('tagger.cli.sys.stderr') as mock_stderr:
                writer = run.TqdmWriter()
                writer.write("test message")
                mock_stderr.write.assert_called_once_with("test message")

    def test_tqdm_writer_with_progress_bar(self):
        """Test TqdmWriter when progress bar is active."""
        mock_progress_bar = Mock()
        with patch('tagger.cli._current_progress_bar', mock_progress_bar):
            with patch('tagger.cli.tqdm.write') as mock_tqdm_write:
                writer = run.TqdmWriter()
                writer.write("test message\n")
                mock_tqdm_write.assert_called_once_with("test message", file=sys.stderr)

    def test_tqdm_writer_flush_no_progress_bar(self):
        """Test TqdmWriter flush when no progress bar is active."""
        with patch('tagger.cli._current_progress_bar', None):
            with patch('tagger.cli.sys.stderr') as mock_stderr:
                writer = run.TqdmWriter()
                writer.flush()
                mock_stderr.flush.assert_called_once()

    def test_tqdm_writer_flush_with_progress_bar(self):
        """Test TqdmWriter flush when progress bar is active."""
        mock_progress_bar = Mock()
        with patch('tagger.cli._current_progress_bar', mock_progress_bar):
            with patch('tagger.cli.sys.stderr') as mock_stderr:
                writer = run.TqdmWriter()
                writer.flush()
                mock_stderr.flush.assert_not_called()