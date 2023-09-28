import os
import re

import torch
import numpy as np
from pathlib import Path

import tagger.deepbooru_model as deepbooru_model

from tagger.image import resize_image
import tagger.modelloader as modelloader
import contextlib

model_path = Path("./models")

interrogate_keep_models_in_memory = False
interrogate_deepbooru_score_threshold = 0.5
deepbooru_use_spaces = True
deepbooru_escape = True
deepbooru_sort_alpha = True
interrogate_return_ranks = False 
deepbooru_filter_tags = ""

dtype = torch.float64
device_cpu = torch.device('cpu')
device_main = torch.device('cpu')

def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32:
        return contextlib.nullcontext()

    return torch.autocast("cuda")

def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        from torch.mps import empty_cache
        empty_cache()

re_special = re.compile(r'([\\()])')

class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return

        files = modelloader.load_models(
            model_path=os.path.join(model_path, "torch_deepdanbooru"),
            model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
            ext_filter=[".pt"],
            download_name='model-resnet_custom_v3.pt',
        )

        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(files[0], map_location="cpu"))

        self.model.eval()
        self.model.to(device_cpu, dtype)

    def start(self):
        self.load()
        self.model.to(device_main)

    def stop(self):
        if not interrogate_keep_models_in_memory:
            self.model.to(device_cpu)
            torch_gc()

    def tag(self, pil_image):
        self.start()
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(self, pil_image, force_disable_ranks=False):
        threshold = interrogate_deepbooru_score_threshold
        use_spaces = deepbooru_use_spaces
        use_escape = deepbooru_escape
        alpha_sort = deepbooru_sort_alpha
        include_ranks = interrogate_return_ranks and not force_disable_ranks

        pic = resize_image(2, pil_image.convert("RGB"), 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), autocast():
            x = torch.from_numpy(a).to(device_main)
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        filtertags = {x.strip().replace(' ', '_') for x in deepbooru_filter_tags.split(",")}

        for tag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)

model = DeepDanbooru()