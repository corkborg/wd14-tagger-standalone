"""
Interrogator for the Camie-Tagger model.
reference:
https://huggingface.co/Camais03/camie-tagger/blob/main/onnx_inference.py
"""
import sys
import json
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from tagger.interrogator.interrogator import AbsInterrogator

class CamieTaggerInterrogator(AbsInterrogator):
    repo_id: str
    model_path: str
    tags_path: str

    def __init__(
        self,
        name: str,
        repo_id: str,
        model_path: str,
        tags_path='metadata.json',
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.repo_id = repo_id
        self.tags = None
        self.model = None

    def download(self) -> tuple[str, str]:
        print(f"Loading {self.name} model file from {self.repo_id}", file=sys.stderr)

        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.model_path
        )
        tags_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.tags_path,
        )
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        self.model = InferenceSession(model_path,
                                        providers=self.providers)
        print(f'Loaded {self.name} model from {model_path}', file=sys.stderr)

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.metadata = json.load(filen)

    def interrogate(
        self,
        image: Image.Image
    ) -> tuple[
        dict[str, float],  # rating confidents
        dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            self.load()
        if self.model is None:
            raise Exception("Model not loading.")

        # Use the NumPy preprocessing function
        # Result is already a NumPy array with shape (C, H, W) and float32 type
        img_array_chw = preprocess_image(image) # e.g., shape (3, 512, 512)

        # ONNX models typically expect input in NCHW format (Batch, Channel, Height, Width)
        # Add a batch dimension (N=1) at the beginning
        img_numpy = np.expand_dims(img_array_chw, axis=0) # Now shape (1, 3, 512, 512)

        input_ = self.model.get_inputs()[0]

        # Ensure input data type matches model expectation (usually float32)
        if input_.type == 'tensor(float)':
            img_numpy = img_numpy.astype(np.float32)

        # evaluate model
        outputs = self.model.run(None, {input_.name: img_numpy})

        # Process outputs
        initial_probs: np.ndarray = 1.0 / (1.0 + np.exp(-outputs[0]))  # Apply sigmoid
        refined_probs: np.ndarray = 1.0 / (1.0 + np.exp(-outputs[1])) if len(outputs) > 1 else initial_probs

        # Get top tags
        indices = np.atleast_1d(refined_probs[0]).nonzero()[0]

        # Group by category
        tags_by_category = {}
        for idx in indices:
            idx_str = str(idx)
            tag_name = self.metadata['idx_to_tag'].get(idx_str, f"unknown-{idx}")
            category = self.metadata['tag_to_category'].get(tag_name, "general")

            if category not in tags_by_category:
                tags_by_category[category] = []

            prob = float(refined_probs[0, idx])
            tags_by_category[category].append((tag_name, prob))

        # 'year', 'rating', 'general', 'character', 'copyright', 'artist', 'meta'
        rating_tags = dict(tags_by_category.get('rating', []))
        general_tags = dict(tags_by_category.get('general', []))

        return rating_tags, general_tags

def preprocess_image(img: Image.Image, image_size: int = 512) -> np.ndarray:
    """
    Process a PIL image for inference using NumPy.

    Args:
        img: The input PIL Image.
        image_size: The target square size (width and height) for the output array.

    Returns:
        A NumPy array representing the processed image in CHW format,
        with pixel values scaled to [0.0, 1.0] and dtype float32.
    """

    # Convert to RGB
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # Get original dimensions
    width, height = img.size
    aspect_ratio = width / height

    # Calculate new dimensions to maintain aspect ratio
    if aspect_ratio > 1:
        new_width = image_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = image_size
        new_width = int(new_height * aspect_ratio)

    # Resize with LANCZOS filter
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with padding
    new_image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2
    new_image.paste(img, (paste_x, paste_y))

    # Convert PIL image to NumPy array (H, W, C) with values 0-255
    img_array = np.array(new_image, dtype=np.float32)

    # Scale pixel values from [0, 255] to [0.0, 1.0]
    img_array /= 255.0

    # Change dimension order from HWC to CHW
    img_array = img_array.transpose((2, 0, 1)) # C, H, W

    return img_array
