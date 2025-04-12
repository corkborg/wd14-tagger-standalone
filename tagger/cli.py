import sys
from typing import Generator, Iterable
from .interrogator.interrogator import AbsInterrogator # Relative import
from PIL import Image
from pathlib import Path
import argparse

from .interrogators import interrogators # Relative import

def main():
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
        '--overwrite',
        action='store_true',
        help='Overwrite caption file if it exists')
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU only')
    parser.add_argument(
        '--rawtag',
        action='store_true',
        help='Use the raw output of the model')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Enable recursive file search')
    parser.add_argument(
        '--exclude-tag',
        dest='exclude_tags',
        action='append',
        metavar='t1,t2,t3',
        help='Specify tags to exclude (Need comma-separated list)')
    parser.add_argument(
        '--model',
        default='wd14-convnextv2.v1',
        metavar='MODELNAME',
        help='modelname to use for prediction (default is wd14-convnextv2.v1)')
    args = parser.parse_args()

    # get interrogator configs
    # Use relative import path within the main function or pass interrogators
    # from tagger.interrogators import interrogators # This needs adjustment
    interrogator = interrogators[args.model] 

    if args.cpu:
        interrogator.use_cpu()

    def parse_exclude_tags() -> set[str]:
        if args.exclude_tags is None:
            return set()

        tags = []
        for str_val in args.exclude_tags: # Renamed str to str_val to avoid shadowing built-in
            for tag in str_val.split(','):
                tags.append(tag.strip())

        # reverse escape (nai tag to danbooru tag)
        reverse_escaped_tags = []
        for tag in tags:
            tag = tag.replace(' ', '_').replace(r'\(', '(').replace(r'\)', ')')
            reverse_escaped_tags.append(tag)
        return set([*tags, *reverse_escaped_tags])  # reduce duplicates

    def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]) -> dict[str, float]:
        """
        Predictions from a image path
        """
        im = Image.open(image_path)
        result = interrogator.interrogate(im)

        # Use relative import for AbsInterrogator or adjust access
        # from .interrogator.interrogator import AbsInterrogator # This needs adjustment
        return AbsInterrogator.postprocess_tags(
            result[1],
            threshold=args.threshold,
            escape_tag=tag_escape,
            replace_underscore=tag_escape,
            exclude_tags=exclude_tags)

    def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
        """
        Explore files by folder path
        """
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
                yield path
            elif args.recursive and path.is_dir():
                yield from explore_image_files(path)

    if args.dir:
        root_path = Path(args.dir)
        for image_path in explore_image_files(root_path):
            caption_path = image_path.parent / f'{image_path.stem}{args.ext}'

            if caption_path.is_file() and not args.overwrite:
                # skip if caption exists
                print('skip:', image_path)
                continue

            print('processing:', image_path)
            tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags())

            tags_str = ', '.join(tags.keys())

            with open(caption_path, 'w') as fp:
                fp.write(tags_str)

    if args.file:
        tags = image_interrogate(Path(args.file), not args.rawtag, parse_exclude_tags())
        print(file=sys.stderr)
        tags_str = ', '.join(tags.keys())
        print(tags_str)

# Note: The if __name__ == "__main__" block is typically not needed
# for files intended to be used as modules/entry points like this.
# It doesn't hurt, but it won't be executed when called via the entry point.
if __name__ == "__main__":
    main() 