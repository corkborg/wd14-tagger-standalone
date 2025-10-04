import sys
import json
import logging
import warnings
from tqdm import tqdm
from typing import Generator, Iterable
from tagger.interrogator.interrogator import AbsInterrogator
from PIL import Image, ImageFile
from pathlib import Path
from typing import Dict
import argparse
import structlog

from tagger.interrogators import interrogators

# Allow images with broken headers to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Filter CUDA provider warning
warnings.filterwarnings('ignore', message='Specified provider .* is not in available provider names')

# Global progress bar reference for tqdm-safe logging
_current_progress_bar = None

class TqdmWriter:
    """Custom writer that uses tqdm.write() if progress bar is active"""
    def write(self, message):
        if _current_progress_bar is not None:
            tqdm.write(message.rstrip(), file=sys.stderr)
        else:
            sys.stderr.write(message)

    def flush(self):
        if _current_progress_bar is None:
            sys.stderr.flush()

# Configure structured logging with tqdm compatibility
def setup_logging(enable_progress_bar: bool = False):
    # Create custom handler that uses TqdmWriter
    handler = logging.StreamHandler(TqdmWriter())
    handler.setLevel(logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    return structlog.get_logger("wd14-tagger")

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
    '--additional-tag',
    dest='additional_tags',
    action='append',
    metavar='t1,t2,t3',
    help='Specify additional tags (Need comma-separated list)')
parser.add_argument(
    '--model',
    default='wd14-convnextv2.v1',
    metavar='MODELNAME',
    help='modelname to use for prediction (default is wd14-convnextv2.v1)')
parser.add_argument(
    '--json',
    action='store_true',
    help='output json instead of plaintext'
)
parser.add_argument(
    '--progress-bar',
    action='store_true',
    help='Show progress bar instead of processing messages'
)
def parse_args():
    """Parse command line arguments."""
    return parser.parse_args()


def process_directory(args, logger, interrogator):
    """Process all images in a directory."""
    global _current_progress_bar

    root_path = Path(args.dir)
    # First collect all files to process for progress bar and logging
    image_files = list(explore_image_files(root_path, args.recursive))

    if not image_files:
        logger.warning("No image files found", directory=str(root_path))
        return

    logger.info("Starting batch processing",
                directory=str(root_path),
                total_files=len(image_files),
                model=args.model,
                threshold=args.threshold)

    if args.progress_bar:
        _current_progress_bar = tqdm(image_files, desc="Processing images")
        iterator = _current_progress_bar
    else:
        iterator = image_files

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in iterator:
        caption_path = image_path.parent / f'{image_path.stem}{args.ext}'

        if caption_path.is_file() and not args.overwrite:
            skipped_count += 1
            if not args.progress_bar:
                logger.info("Skipping existing file",
                           file=str(image_path.name),
                           caption_exists=True)
            continue

        if not args.progress_bar:
            logger.info("Processing image",
                       file=str(image_path.name),
                       path=str(image_path))
        try:
            tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags(args), parse_additional_tags(args), args, interrogator)
            with open(caption_path, 'w') as fp:
                fp.write(generate_output_string(image_path, tags, args))
            processed_count += 1

            if not args.progress_bar:
                logger.info("Generated tags",
                           file=str(image_path.name),
                           tag_count=len(tags),
                           output_file=str(caption_path.name))
        except Exception as e:
            error_count += 1
            logger.error("Processing failed",
                        file=str(image_path.name),
                        path=str(image_path),
                        error=str(e),
                        error_type=type(e).__name__)

    # Clean up progress bar
    if _current_progress_bar:
        _current_progress_bar.close()
        _current_progress_bar = None

    logger.info("Batch processing complete",
                processed=processed_count,
                skipped=skipped_count,
                errors=error_count,
                total=len(image_files))


def process_single_file(args, logger, interrogator):
    """Process a single image file."""
    file_path = Path(args.file)
    logger.info("Processing single file",
                file=str(file_path.name),
                path=str(file_path),
                model=args.model,
                threshold=args.threshold)
    try:
        tags = image_interrogate(file_path, not args.rawtag, parse_exclude_tags(args), parse_additional_tags(args), args, interrogator)
        output = generate_output_string(args.file, tags, args)
        print(output)
        logger.info("Generated tags for single file",
                   file=str(file_path.name),
                   tag_count=len(tags),
                   output_format="json" if args.json else "text")
    except Exception as e:
        logger.error("Single file processing failed",
                    file=str(file_path.name),
                    path=str(file_path),
                    error=str(e),
                    error_type=type(e).__name__)
        sys.exit(1)

def parse_exclude_tags(args) -> set[str]:
    if args.exclude_tags is None:
        return set()

    tags = []
    for str in args.exclude_tags:
        for tag in str.split(','):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(' ', '_').replace(r'\(', '(').replace(r'\)', ')')
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates

def parse_additional_tags(args) -> list[str]:
    if args.additional_tags is None:
        return list()

    tags = []
    for str in args.additional_tags:
        for tag in str.split(','):
            tags.append(tag.strip())
    return list(set(tags))

def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str], additional_tags: list[str], args, interrogator) -> dict[str, float]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    return AbsInterrogator.postprocess_tags(
        result[1],
        threshold=args.threshold,
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags,
        additional_tags=additional_tags)

def explore_image_files(folder_path: Path, recursive: bool) -> Generator[Path, None, None]:
    """
    Explore files by folder path in lexicographic order
    """
    paths = []
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
            paths.append(path)
        elif recursive and path.is_dir():
            paths.extend(explore_image_files(path, recursive))
    paths.sort()
    yield from paths

def generate_output_string(image_path: str, tags: Dict, args):
    if args.json:
        return json.dumps({"file":str(image_path),"tags":tags})
    else:
        return ', '.join(tags.keys())



def main():
    """Main entry point for the WD14 tagger."""
    args = parse_args()

    # Initialize structured logging
    logger = setup_logging(args.progress_bar)

    # get interrogator configs
    interrogator = interrogators[args.model]

    if args.cpu:
        interrogator.use_cpu()

    # Set quiet mode if progress bar is enabled
    if args.progress_bar:
        interrogator.set_quiet(True)
        logging.getLogger('tagger.interrogator').setLevel(logging.WARNING)

    # Process based on arguments
    if args.dir:
        process_directory(args, logger, interrogator)
    elif args.file:
        process_single_file(args, logger, interrogator)


if __name__ == "__main__":
    main()
