import json
import argparse
from pathlib import Path
import sys
from enum import Enum

class OutputFormat(Enum):
    DENSE_VECTOR = "dense-vector"
    QDRANT_DENSE = "qdrant-dense-vector"
    QDRANT_SPARSE = "qdrant-sparse-vector"

    def __str__(self):
        return self.value

VALID_CATEGORIES = {'year', 'character', 'artist', 'meta', 'copyright', 'general', 'rating'}

def validate_categories(categories):
    invalid = set(categories) - VALID_CATEGORIES
    if invalid:
        print(f"Error: Invalid categories: {', '.join(invalid)}", file=sys.stderr)
        print(f"Valid categories are: {', '.join(sorted(VALID_CATEGORIES))}", file=sys.stderr)
        sys.exit(1)

def load_reference_mapping(mapping_path):
    with open(mapping_path) as f:
        return json.load(f)

def create_compact_mapping(ref_data, categories, is_exclude=True):
    valid_tags = []
    idx_to_tag = ref_data['idx_to_tag']
    tag_to_category = ref_data['tag_to_category']

    for idx in range(len(idx_to_tag)):
        tag = idx_to_tag[str(idx)]
        category = tag_to_category[tag]
        # If excluding, keep tags NOT in categories
        # If including, keep tags IN categories
        if (is_exclude and category not in categories) or (not is_exclude and category in categories):
            valid_tags.append(tag)

    tag_to_compact_idx = {tag: idx for idx, tag in enumerate(valid_tags)}
    return tag_to_compact_idx, len(valid_tags)

def convert_to_dense_array(input_tags, tag_to_compact_idx, array_size):
    result = [0.0] * array_size
    for tag, confidence in input_tags.items():
        if tag in tag_to_compact_idx:
            result[tag_to_compact_idx[tag]] = confidence
    return result

def format_output(dense_array, tag_to_compact_idx, output_format, original_data):
    if output_format == OutputFormat.DENSE_VECTOR:
        return json.dumps(dense_array)

    elif output_format == OutputFormat.QDRANT_DENSE:
        return json.dumps({
            "vector": dense_array,
            "payload": original_data  # Include complete original data
        })

    elif output_format == OutputFormat.QDRANT_SPARSE:
        # Convert dense to sparse format
        non_zero = [(idx, val) for idx, val in enumerate(dense_array) if val > 0]
        indices, values = zip(*non_zero) if non_zero else ([], [])
        return json.dumps({
            "vector": {
                "indices": list(indices),
                "values": list(values)
            },
            "payload": original_data  # Include complete original data
        })

def parse_output_format(value):
    try:
        return OutputFormat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid output format. Valid options are: {', '.join(f.value for f in OutputFormat)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input JSON file with tags and confidences')
  
    # Create mutually exclusive group
    category_group = parser.add_mutually_exclusive_group()
    category_group.add_argument('--exclude-category',
                              type=str,
                              help=f'Comma-separated list of categories to exclude. Valid categories: {", ".join(sorted(VALID_CATEGORIES))}')
    category_group.add_argument('--include-category',
                              type=str,
                              help=f'Comma-separated list of categories to include. Valid categories: {", ".join(sorted(VALID_CATEGORIES))}')
  
    parser.add_argument('--tag-mapping',
                       type=str,
                       default=str(Path.home() / '.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/c7dc4e38696a812e593916e3f2e51b92f687f8ea'),
                       help='Path to tag mapping JSON file')
    parser.add_argument(
        '--output-format',
        type=parse_output_format,
        default=OutputFormat.DENSE_VECTOR,
        choices=list(OutputFormat),
        help=f'Output format type. Options: {", ".join(f.value for f in OutputFormat)}'
    )

    args = parser.parse_args()

    # Parse and validate categories
    categories = set()
    is_exclude = True
    if args.exclude_category:
        categories = set(args.exclude_category.split(','))
        is_exclude = True
    elif args.include_category:
        categories = set(args.include_category.split(','))
        is_exclude = False
  
    validate_categories(categories)

    # Load reference data
    ref_data = load_reference_mapping(args.tag_mapping)

    # Create compact mapping
    tag_to_compact_idx, array_size = create_compact_mapping(ref_data, categories, is_exclude)

    # Load input data
    with open(args.input_file) as f:
        input_data = json.load(f)
        input_tags = input_data['tags']  # Get tags from new structure
      
    # Create compact mapping and dense array
    tag_to_compact_idx, array_size = create_compact_mapping(ref_data, categories, is_exclude)
    dense_array = convert_to_dense_array(input_tags, tag_to_compact_idx, array_size)

    # Format and output
    print(format_output(dense_array, tag_to_compact_idx, args.output_format, input_data))

if __name__ == '__main__':
    main()
