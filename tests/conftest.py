"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 64x64 RGB image
    image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(image_array, mode='RGB')


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA PIL Image for testing."""
    # Create a simple 64x64 RGBA image
    image_array = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    return Image.fromarray(image_array, mode='RGBA')


@pytest.fixture
def mock_onnx_session():
    """Create a mock ONNX InferenceSession."""
    mock_session = Mock()

    # Mock input specification
    mock_input = Mock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 224, 224]
    mock_session.get_inputs.return_value = [mock_input]

    # Mock output specification
    mock_output = Mock()
    mock_output.name = "output"
    mock_session.get_outputs.return_value = [mock_output]

    # Mock inference results
    mock_session.run.return_value = [np.random.rand(1, 1000).astype(np.float32)]

    return mock_session


@pytest.fixture
def mock_tags_data():
    """Create mock tags data for testing."""
    return {
        "tags": ["tag1", "tag2", "tag3", "explicit_tag"],
        "ratings": ["general", "sensitive", "questionable", "explicit"]
    }


@pytest.fixture
def sample_file_path(tmp_path):
    """Create a temporary file path for testing."""
    return tmp_path / "test_image.jpg"


@pytest.fixture
def mock_logger():
    """Create a mock structlog logger."""
    return MagicMock()