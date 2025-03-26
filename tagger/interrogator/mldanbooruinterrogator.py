import sys
import json

from PIL import Image
from numpy import asarray, float32, expand_dims, exp

from huggingface_hub import hf_hub_download

from tagger.interrogator import AbsInterrogator
import tagger.dbimutils as dbimutils


class MLDanbooruInterrogator(AbsInterrogator):
    """ Interrogator for the MLDanbooru model. """
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

        from onnxruntime import InferenceSession
        self.model = InferenceSession(model_path,
                                        providers=self.providers)
        print(f'Loaded {self.name} model from {model_path}', file=sys.stderr)

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.tags = json.load(filen)

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

        if self.tags is None:
            raise Exception("Tags not loading.")

        tags = {str(tag): float(conf) for tag, conf in zip(self.tags, y.flatten())}
        return {}, tags

    def large_batch_interrogate(self, images: list, dry_run=False) -> str:
        raise NotImplementedError()

