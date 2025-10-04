
# WD14 Tagger Standalone

Standalone WD14 tagger for image classification using ONNX models.

Forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)

## Installation

### Package Installation (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Development Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

### Command Line Tool

After installation, use the `wd14-tagger` command:

```bash
# Single file
wd14-tagger --file image.jpg

# Batch processing
wd14-tagger --dir /path/to/images

# With options
wd14-tagger --file image.jpg --threshold 0.5 --json --progress-bar
```

### Legacy Script (Backward Compatibility)

You can still use the original script:

```bash
python run.py --file image.jpg
python run.py --dir /path/to/images
```

### Options

```
usage: wd14-tagger [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD]
                   [--ext EXT] [--overwrite] [--cpu] [--rawtag] [--recursive]
                   [--exclude-tag t1,t2,t3] [--additional-tag t1,t2,t3]
                   [--model MODELNAME] [--json] [--progress-bar]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option
                        (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only
  --rawtag              Use the raw output of the model
  --recursive           Enable recursive file search
  --exclude-tag t1,t2,t3
                        Specify tags to exclude (Need comma-separated list)
  --additional-tag t1,t2,t3
                        Specify additional tags (Need comma-separated list)
  --model MODELNAME     modelname to use for prediction (default is
                        wd14-convnextv2.v1)
  --json                output json instead of plaintext
  --progress-bar        Show progress bar instead of processing messages
```

## Supported Models

### Using the wd14-tagger command:

```bash
# Camie Tagger (released 2025)
wd14-tagger --model camie-tagger --file image.jpg

# SmilingWolf large model. (released 2024)
wd14-tagger --model wd-vit-large-tagger-v3 --file image.jpg
wd14-tagger --model wd-eva02-large-tagger-v3 --file image.jpg

# SmilingWolf v3 model. (released 2024)
wd14-tagger --model wd-v1-4-vit-tagger.v3 --file image.jpg
wd14-tagger --model wd-v1-4-convnext-tagger.v3 --file image.jpg
wd14-tagger --model wd-v1-4-swinv2-tagger.v3 --file image.jpg

# SmilingWolf v2 model. (released 2023)
wd14-tagger --model wd-v1-4-moat-tagger.v2 --file image.jpg
wd14-tagger --model wd14-vit.v2 --file image.jpg
wd14-tagger --model wd14-convnext.v2 --file image.jpg

# SmilingWolf v1 model. (released 2022)
wd14-tagger --model wd14-vit.v1 --file image.jpg
wd14-tagger --model wd14-convnext.v1 --file image.jpg
wd14-tagger --model wd14-convnextv2.v1 --file image.jpg
wd14-tagger --model wd14-swinv2-v1 --file image.jpg

# Z3D-E621-Convnext
wd14-tagger --model z3d-e621-convnext-toynya --file image.jpg
wd14-tagger --model z3d-e621-convnext-silveroxides --file image.jpg

# kiriyamaX model.
wd14-tagger --model mld-caformer.dec-5-97527 --file image.jpg
wd14-tagger --model mld-tresnetd.6-30000 --file image.jpg
```

### Legacy script usage (all models also work with `python run.py`):

```bash
python run.py --model camie-tagger --file image.jpg
python run.py --model wd14-convnextv2.v1 --file image.jpg
# ... etc
```

## Using GPU

Requires CUDA 12.2 and cuDNN8.x.

```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

https://onnxruntime.ai/docs/install/</br>
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)
