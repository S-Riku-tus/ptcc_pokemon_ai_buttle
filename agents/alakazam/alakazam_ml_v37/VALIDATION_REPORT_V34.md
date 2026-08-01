# v34 検証報告

## データと分離

- 教師: Yushin Ito
- 元ログ: 1,000試合
- 有効系列: 999
- 全判断: 49,590
- 全意味候補: 443,714
- 特徴: 674
- 分離: episode ID昇順70% / 10% / 20%
- train: 35,704判断 / 309,362候補
- validation: 5,038判断 / 51,311候補
- test: 8,848判断 / 83,041候補

## 採用モデル

- LightGBM LambdaRank
- 255 leaves
- 312 trees
- recency floor 0.25、power 2.0
- 方策状態: 直前4判断、累積action、同一ターン位置、行動turn gap

## 精度

| 指標 | Validation | Test |
|---|---:|---:|
| Top-1意味的一致 | 80.81% | **80.63%** |
| Top-2包含 | 93.27% | **93.30%** |
| Top-3包含 | 97.16% | **96.99%** |

目標Top-1 90%は未達です。

## 研究selector

| 構成 | Validation Top-1 | Test Top-1 | 採用 |
|---|---:|---:|---|
| Top-2 selector | 81.26% | 80.99% | いいえ |
| OOF residual ranker | 80.94% | 81.09% | いいえ |
| leak-free GRU selector | 81.38% | 81.01% | いいえ |

研究selectorは単木を小幅に上回る場合がありますが、validation/testの安定性、標準ライブラリ制約、実装複雑性を考慮して提出本体へ入れていません。

## 実装検証

- Python構文: 成功
- pytest: **181成功 / 0失敗**
- デッキ: 60枚
- v34モデル: 312木 / 674特徴
- LightGBM→JSONスコア一致: 512候補、最大絶対誤差0.0
- 学習特徴→runtime特徴一致: 3 episode、1,251候補、最大絶対誤差0.0
- v31/v32 safety fallback、v29 residual、teacher memory: 継承
- 不採用PyTorch selector: runtimeへ未同梱

## 未実施

- 公式`cg`を使った実エンジン対戦
- Kaggle Validation Episode
- v31/v32/v34のseat-swap Champion–Challenger
- 実ラダーRating

これらは今回の添付ZIPに公式`cg`と対戦ハーネスが含まれていないため未実施です。
