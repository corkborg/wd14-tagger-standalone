

import sys
from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path

interrogator = WaifuDiffusionInterrogator(
    'wd14-convnextv2-v2',
    repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    revision='v2.0'
)

image_path = Path(sys.argv[1])

im = Image.open(image_path)
result = interrogator.interrogate(im)
tags = Interrogator.postprocess_tags(result[1], threshold=0.35)
print()

for k, v in tags.items():
    print(k, v)
print(result[0])

