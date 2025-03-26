"""
Interrogator for the Camie-Tagger model.
reference:
https://huggingface.co/Camais03/camie-tagger/blob/main/onnx_inference.py
"""
import sys
import json

from typing import cast

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from tagger.interrogator import AbsInterrogator

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

        img_tensor = preprocess_image(image)
        img_numpy = img_tensor.unsqueeze(0).numpy()

        input_ = self.model.get_inputs()[0]

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
        return dict(tags_by_category['rating']), dict(tags_by_category['general'])

def preprocess_image(img: Image.Image, image_size=512) -> torch.Tensor:
    """Process an image for inference"""

    # Convert RGBA or Palette images to RGB
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

    import torchvision.transforms as transforms

    # Initialize transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply transforms
    img_tensor= transform(new_image)
    img_tensor = cast(torch.Tensor, img_tensor)
    return img_tensor
