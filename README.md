
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger]()


## install

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

```
usage: run.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD] [--ext EXT]
              [--model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}]
              [--overwrite OVERWRITE]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option (default is .txt)
  --model {wd14-vit.v1,wd14-vit.v2,wd14-convnext.v1,wd14-convnext.v2,wd14-convnextv2.v1,wd14-swinv2-v1,wd-v1-4-moat-tagger.v2,mld-caformer.dec-5-97527,mld-tresnetd.6-30000}
                        modelname to use for prediction (default is wd14-convnextv2.v1)
  --overwrite OVERWRITE
                        Overwrite the file if it exists (default is True)
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
python run.py --file image.jpg --model wd14-vit.v1
python run.py --file image.jpg --model wd14-vit.v2
python run.py --file image.jpg --model wd14-convnext.v1
python run.py --file image.jpg --model wd14-convnext.v2
python run.py --file image.jpg --model wd14-convnextv2.v1
python run.py --file image.jpg --model wd14-swinv2-v1
python run.py --file image.jpg --model wd-v1-4-moat-tagger.v2
python run.py --file image.jpg --model mld-caformer.dec-5-97527
python run.py --file image.jpg --model mld-tresnetd.6-30000
```