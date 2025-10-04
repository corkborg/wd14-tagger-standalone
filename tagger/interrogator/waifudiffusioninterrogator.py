import os
import sys
import logging
import structlog
import pandas as pd
import numpy as np

from typing import Tuple, Dict
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download
import re

tag_escape_pattern = re.compile(r'([\\()])')

from tagger.interrogator.interrogator import AbsInterrogator
import tagger.dbimutils as dbimutils

logger = structlog.get_logger(__name__)

class WaifuDiffusionInterrogator(AbsInterrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        if not self.quiet:
            logger.info("Loading model file", model=self.name, repo_id=self.kwargs['repo_id'])

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        from onnxruntime import InferenceSession
        self.model = InferenceSession(str(model_path), providers=self.providers)

        if not self.quiet:
            logger.info("Model loaded", model=self.name, path=str(model_path))

        self.tags = pd.read_csv(tags_path)

    def interrogate(
        self,
        input_image: Image.Image
    ) -> tuple[
        dict[str, float],  # rating confidents
        dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        if self.model is None:
            raise Exception("Model not loading.")

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = input_image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        if self.tags is None:
            raise Exception("Tags not loading.")

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags
