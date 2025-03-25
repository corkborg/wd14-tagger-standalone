
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)

## install

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

```
usage: run.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD] [--ext EXT] [--overwrite] [--cpu] [--rawtag] [--recursive] [--exclude-tag t1,t2,t3]
              [--model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,wd-v1-4-vit-tagger.v3,wd-v1-4-convnext-tagger.v3,wd-v1-4-swinv2-tagger.v3,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only
  --rawtag              Use the raw output of the model
  --recursive           Enable recursive file search
  --exclude-tag t1,t2,t3
                        Specify tags to exclude (Need comma-separated list)
  --model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,wd-v1-4-vit-tagger.v3,wd-v1-4-convnext-tagger.v3,wd-v1-4-swinv2-tagger.v3,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}
                        modelname to use for prediction (default is wd14-convnextv2.v1)
```

single file

```
python run.py --file image.jpg
```

batch execution

```
python run.py --dir dir/dir
```

## Support Models

```
# SmilingWolf large model. (released 2024)
python run.py --file image.jpg --model wd-vit-large-tagger-v3
python run.py --file image.jpg --model wd-eva02-large-tagger-v3

# SmilingWolf v3 model. (released 2024)
python run.py --file image.jpg --model wd-v1-4-vit-tagger.v3
python run.py --file image.jpg --model wd-v1-4-convnext-tagger.v3
python run.py --file image.jpg --model wd-v1-4-swinv2-tagger.v3

# SmilingWolf v2 model. (released 2023)
python run.py --file image.jpg --model wd-v1-4-moat-tagger.v2
python run.py --file image.jpg --model wd14-vit.v2
python run.py --file image.jpg --model wd14-convnext.v2

# SmilingWolf v1 model. (released 2022)
python run.py --file image.jpg --model wd14-vit.v1
python run.py --file image.jpg --model wd14-convnext.v1
python run.py --file image.jpg --model wd14-convnextv2.v1
python run.py --file image.jpg --model wd14-swinv2-v1

# Z3D-E621-Convnext
python run.py --file image.jpg --model z3d-e621-convnext-toynya
python run.py --file image.jpg --model z3d-e621-convnext-silveroxides

# kiriyamaX model.
python run.py --file image.jpg --model mld-caformer.dec-5-97527
python run.py --file image.jpg --model mld-tresnetd.6-30000
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
