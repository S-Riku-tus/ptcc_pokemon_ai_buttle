# v4 検証報告

## データ分離

- 最終評価教師: Yushin Ito
- 主教師: 999episode / 49,590判断
- 分離: episode ID順70% / 10% / 20%
- train: 35,704判断
- validation: 5,038判断
- test: 8,848判断
- 追加教師は既存testより前の別episodeのみ
- blend係数とthresholdはvalidationで固定
- testは構成固定後に評価

## 追加教師

- Majkel1337: 999episode / 53,782判断
- Yushin Ito 54773249: 176episode / 9,397判断
- Rmy: 699episode / 29,871判断

全てを単純混合せず、validationで汎化改善を確認したRmy expert 15% blendだけを採用した。

## 精度

| 構成 | Validation | Test |
|---|---:|---:|
| v1 | 80.81% | 80.63% |
| v2 | — | 81.60% |
| v3 | 81.70% | 81.62% |
| **v4** | **81.94%** | **81.78%** |
| 目標 | — | 90.00% |

- v4 test正解: 7,236 / 8,848
- v3との差: +14判断
- 改善: +0.158ポイント
- v4行動種精度: 84.96%

## compact runtime一致

500 test判断でnative XGBoostとcompact純Pythonを比較した。

- primary model最大確率誤差: 6.09e-7
- Rmy expert最大確率誤差: 5.57e-7
- blend最大確率誤差: 5.05e-7
- blend argmax不一致: 0
- Rmy expert compact trees: 1,661
- primary compact trees: 2,497
- type ensemble推論: 約5.83ms / 判断（100判断測定）

## 静的・回帰検証

- Python構文検査: 成功
- pytest: 187成功 / 0失敗
- v4新規テスト: 3成功
- デッキ: 60枚
- 外部MLライブラリ: runtimeでは不使用
- `rmy_type_model.json`読込: 成功
- primary / expert classes・selected columns一致: 成功

## 採否

- 90%目標: 未達
- v3よりvalidationとtestの双方で改善: はい
- v4をchallengerとして保存: はい
- v31を実戦championとして維持: はい
