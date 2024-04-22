

import sys
from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path
import argparse

from tagger.interrogators import interrogators

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dir', help='Predictions for all images in the directory')
group.add_argument('--file', help='Predictions for one file')

parser.add_argument(
    '--threshold',
    type=float,
    default=0.35,
    help='Prediction threshold (default is 0.35)')
parser.add_argument(
    '--ext',
    default='.txt',
    help='Extension to add to caption file in case of dir option (default is .txt)')
parser.add_argument(
    '--model',
    default='wd14-convnextv2.v1',
    choices=list(interrogators.keys()),
    help='modelname to use for prediction (default is wd14-convnextv2.v1)')
parser.add_argument(
    '--overwrite',
    default=True,
    type=bool,
    help='Overwrite the file if it exists (default is True)')
args = parser.parse_args()

# get interrogator configs
interrogator = interrogators[args.model]

def image_interrogate(image_path: Path):
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)
    return Interrogator.postprocess_tags(result[1], threshold=args.threshold)

if args.dir:
    d = Path(args.dir)
    for f in d.iterdir():
        if not f.is_file() or f.suffix not in ['.png', '.jpg', '.jpeg', '.webp']:
            continue
        image_path = Path(f)
        print('processing:', image_path)
        tags = image_interrogate(image_path)
        tags_str = ", ".join(tags.keys())
        with open(f.parent / f"{f.stem}{args.ext}", "w") as fp:
            fp.write(tags_str)

if args.file:
    tags = image_interrogate(Path(args.file))
    print()
    tags_str = ", ".join(tags.keys())
    print(tags_str)


