

import sys
from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path
import argparse

from tagger.interrogators import interrogators

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dir', help='ディレクトリ内の画像すべてに予測を行う')
group.add_argument('--file', help='ファイルに対して予測を行う')

parser.add_argument(
    '--threshold',
    type=float,
    default=0.35,
    help='予測値の足切り確率（デフォルトは0.35）')
parser.add_argument(
    '--ext',
    default='.txt',
    help='dirの場合にキャプションファイルにつける拡張子')
parser.add_argument(
    '--model',
    default='wd14-convnextv2.v1',
    choices=list(interrogators.keys()),
    help='予測に使用するモデル名')

args = parser.parse_args()

# 使用するinterrogatorを読み出す
interrogator = interrogators[args.model]

def image_interrogate(image_path: Path):
    """
    画像パスから予測を行う
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)
    return Interrogator.postprocess_tags(result[1], threshold=args.threshold)

if args.dir:
    d = Path(args.dir)
    for f in d.iterdir():
        if not f.is_file() or f.suffix not in ['.png', '.jpg', '.webp']:
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


