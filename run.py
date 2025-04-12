#!/usr/bin/env python
import sys

# This script remains for backward compatibility or direct execution.
# The main CLI logic is now in tagger.cli

if __name__ == "__main__":
    try:
        from tagger.cli import main
    except ImportError as e:
        print("Error: Unable to import the tagger module.", file=sys.stderr)
        print("Please ensure the package is installed correctly (e.g., using 'pip install -e .').", file=sys.stderr)
        print(f"Import error details: {e}", file=sys.stderr)
        sys.exit(1)
    
    main()


