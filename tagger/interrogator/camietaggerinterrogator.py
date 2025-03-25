import os
import sys
import pandas as pd
import numpy as np

from typing import Iterable, Tuple, List, Dict
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download
import re
import json

import torchvision.transforms as transforms

from numpy import asarray, float32, expand_dims, exp

tag_escape_pattern = re.compile(r'([\\()])')

import tagger.dbimutils as dbimutils

class CamieTaggerInterrogator(Interrogator):
    """
    Interrogator for the Camie-Tagger model.
    reference:
    https://huggingface.co/Camais03/camie-tagger/blob/main/onnx_inference.py
    """
    repo_id: str
    model_path: str
    tags_path: str

    def __init__(
        self,
        name: str,
        repo_id: str,
        model_path: str,
        tags_path='classes.json',
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.repo_id = repo_id
        self.tags = None
        self.model = None

    def download(self) -> Tuple[str, str]:
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

        from onnxruntime import InferenceSession
        self.model = InferenceSession(model_path,
                                        providers=self.providers)
        print(f'Loaded {self.name} model from {model_path}', file=sys.stderr)

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.tags = json.load(filen)

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            self.load()

        image = dbimutils.fill_transparent(image)
        image = dbimutils.resize(image, 448)  # TODO CUSTOMIZE

        x = asarray(image, dtype=float32) / 255
        # HWC -> 1CHW
        x = x.transpose((2, 0, 1))
        x = expand_dims(x, 0)

        input_ = self.model.get_inputs()[0]
        output = self.model.get_outputs()[0]
        # evaluate model
        y, = self.model.run([output.name], {input_.name: x})

        # Softmax
        y = 1 / (1 + exp(-y))

        tags = {tag: float(conf) for tag, conf in zip(self.tags, y.flatten())}
        return {}, tags

def preprocess_image(image_path, image_size=512):
    """Process an image for inference"""
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found at path: {image_path}")

    # Initialize transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    try:
        with Image.open(image_path) as img:
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

            # Apply transforms
            img_tensor = transform(new_image)
            return img_tensor
    except Exception as e:
        raise Exception(f"Error processing {image_path}: {str(e)}")
