### 分類学習用コマンド

python dr-superimposer train --input=dr-superimposer/data/data.tsv --model_intent=dr-superimposer/model/model_intent.pth --model_datetime=dr-superimposer/model/model_datetime.pth --model_place=dr-superimposer/model/model_place.pth


### モデルを使って試す

python dr-superimposer eval --model_intent=dr-superimposer/model/model_intent.pth --model_datetime=dr-superimposer/model/model_datetime.pth --model_place=dr-superimposer/model/model_place.pth --text=今日の天気は？