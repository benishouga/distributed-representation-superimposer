### 分類学習用コマンド

python distributed-representation-superimposer train --input=distributed-representation-superimposer/data/data.tsv --model_intent=distributed-representation-superimposer/model/model_intent.pth --model_datetime=distributed-representation-superimposer/model/model_datetime.pth --model_place=distributed-representation-superimposer/model/model_place.pth


### モデルを使って試す

python distributed-representation-superimposer eval --model_intent=distributed-representation-superimposer/model/model_intent.pth --model_datetime=distributed-representation-superimposer/model/model_datetime.pth --model_place=distributed-representation-superimposer/model/model_place.pth --text=今日の天気は？