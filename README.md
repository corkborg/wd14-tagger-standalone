
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger]()


## install

```
venv/bin/python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

## usage

```
usage: run.py [-h] [--dir DIR | --file FILE] [--threthold THRETHOLD] [--ext EXT]

options:
  -h, --help            show this help message and exit
  --dir DIR             ディレクトリ内の画像すべてに予測を行う
  --file FILE           ファイルに対して予測を行う
  --threthold THRETHOLD
                        予測値の足切り確率（デフォルトは0.35）
  --ext EXT             dirの場合にキャプションファイルにつける拡張子
```

single file

```
python3 run.py --file image.jpg
```

batch execution

```
python3 run.py --dir dir/dir
```