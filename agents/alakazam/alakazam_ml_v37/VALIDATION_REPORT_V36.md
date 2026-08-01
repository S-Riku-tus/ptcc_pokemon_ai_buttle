# v36 検証報告

## データ分離

- 教師: Yushin Ito
- 一意episode: 999
- MAIN判断: 49,590
- 意味候補: 443,714
- 分離: episode ID順70% / 10% / 20%
- train: 35,704判断
- validation: 5,038判断
- test: 8,848判断
- 特徴・モデル・threshold選択: validationのみ
- test: 構成固定後に確認

## 精度

| 構成 | Validation | Test | 判定 |
|---|---:|---:|---|
| v34候補ranker Top-1 | 80.81% | 80.63% | 基準 |
| v35行動種分類 | 84.44% | 84.88% | 旧type model |
| v36 rich行動種分類 | **84.64%** | 84.86% | validation改善、test同等 |
| v36 rich階層方策 | **81.70%** | **81.62%** | 採用 |
| 正解行動種内v34 ranker | — | 95.59% | oracle条件 |
| 目標 | — | 90.00% | 未達 |

v36はtestで7,222 / 8,848判断一致し、v35の7,220 / 8,848から2判断増えた。差は+0.02ポイントであり、実質的には同等と判断する。

## runtime一致

- 学習用rich rowとruntime row: 300判断
- row長: 1,052
- 最大絶対特徴誤差: 4.77e-7
- 不一致判断: 0
- XGBoostと純Python compact推論: 500判断
- argmax不一致: 0
- 最大確率誤差: 6.09e-7
- XGBoostと同じfloat32 split比較をruntimeへ実装

## 静的・回帰検証

- Python構文検査: 成功
- pytest: 184成功 / 0失敗
- v36新規テスト: 3成功
- デッキ: 60枚
- type representation: 1,052値
- type model入力: 356特徴
- type model: 2,497木 / 11クラス
- 外部MLライブラリ: runtimeでは不使用

## 採否

- 90%目標: 未達
- v35より明確に優位: いいえ
- v31をchampionとして維持: はい
- v36: 研究・ラダー検証用challenger

既存testはv34・v35研究ですでに観測されており、完全に新しい最終holdoutではない。この点からも、2判断の改善を強い一般化証拠として扱わない。
