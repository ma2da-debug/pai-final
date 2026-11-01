# 言葉で指示できる片付けロボット

Physical AI 応用編1 最終課題。MuJoCo上のFranka Pandaが、`put the red cube in the left bin` のような指示を受け、カメラ画像とCLIPで対象を探して箱へ運びます。つかめたか、箱に入ったかを自分で確認し、失敗すれば観測からやり直します。

## 処理の流れ

1. **指示の解釈** (`src/sorter.py`): 指示を対象の説明と行き先の箱に分ける
2. **切り出し** (`src/perception.py`): 上部RGB-Dカメラの深度から机より高い領域を物体候補とし、位置と把持向きを求める
3. **言語照合** (`src/language.py`): 学習済みCLIPで候補と説明文を照合する(推論のみ)
4. **動作** (`src/robot.py`): 減衰最小二乗法の逆運動学と最小躍度軌道で把持・運搬する
5. **検証と再試行** (`src/sorter.py`): 指の間隔で空振りを、再撮影で配置を確認し、最大3回まで試す

物体の位置はカメラ画像から求めます。シミュレータの真の位置は、配置の生成、外乱、評価の判定だけに使います。

## 実行方法

Python 3.11、Intel Macで確認しました。初回はCLIPをHugging Faceからダウンロードします。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_demo.py video --seed 3 --out results/demo.mp4   # デモ動画
python src/run_demo.py eval --episodes 20                      # 成功率
python src/eval_clip.py                                        # 言い回しごとの正答率
```

## 評価結果

配置を変えた20エピソード、計80指示の結果です(`results/eval.json`)。各エピソードの1指示目は、把持の直前に物体を9cm動かす外乱ありです。

| 項目 | 結果 |
|---|---|
| 指示どおり箱に入った | 80 / 80 |
| うち外乱あり | 20 / 20(すべて2回目で成功) |
| 外乱なしで1回目に成功 | 60 / 60 |
| 成功と誤判定 | 0 |

CLIPの正答率は20配置で測りました(`results/clip_eval.json`)。色と形の直接の指定、`something green` などの色の指定、`the color of a banana` と `the color of fresh grass` は20/20でした。`the color of a ripe tomato` は17/20、`the color of the ocean` は11/20です。

## 限界

理想的なシミュレーションでの結果で、実機での性能は示しません。深度にノイズはなく、物体は単色の単純な形状で、互いに離して配置しています。外乱は決まった距離の移動だけです。指示の文型は固定で、物体が重なる場面や、指示に合う物体がない場面は扱いません。

## 第三者の成果物と謝辞

- ロボットモデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の `franka_emika_panda`(Apache-2.0、`assets/franka_emika_panda/LICENSE`)。`panda.xml` からキーフレームを削除し、`sorter_scene.xml` を追加しました
- CLIP: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- 実装にはAIコーディングツールのClaude Codeを使用しました
