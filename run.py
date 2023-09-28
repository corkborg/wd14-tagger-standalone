

from tagger.interrogator import WaifuDiffusionInterrogator
from PIL import Image

interrogator = WaifuDiffusionInterrogator(
    'wd14-convnextv2-v2',
    repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    revision='v2.0'
)

im = Image.open("image.jpg")
result = interrogator.interrogate(im)
print()

for k, v in result[1].items():
    if v > 0.25:
        print(k, v)

print(result[0])

