

from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image

interrogator = WaifuDiffusionInterrogator(
    'wd14-convnextv2-v2',
    repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    revision='v2.0'
)

im = Image.open("image.jpg")
result = interrogator.interrogate(im)
tags = Interrogator.postprocess_tags(result[1])
print()

for k, v in tags.items():
    print(k, v)
print(result[0])

