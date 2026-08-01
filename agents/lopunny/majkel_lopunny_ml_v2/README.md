# majkel_lopunny_ml_v2

`Majkel1337` submission `55137818` の Mega Lopunny ex / Dudunsparce を模倣するv2です。v1の候補独立LambdaRankに、合法手集合全体を同時に見るDeepSets listwise補正を追加しました。

未知時系列testの非強制semantic exactは、v1の77.13%からv2の77.48%へ0.35ポイント改善しました。目標85%には未達です。既知ログへ全データ再fitモデルを戻した90.44%はランタイム整合性の値であり、未知性能ではありません。

## v2の構成

- 900木LambdaRank: 全select contextの合法候補を順位付け
- 75,809 parameter DeepSets: 単一選択MAINだけをlistwise再順位付け
- blend: `deep_z + 2.0 * ranker_z`、DeepSets confidence 0.20以上
- 200木L1回帰: 可変選択数
- 純Python runtime: LightGBM/torchを提出時に使用しない
- v1と同じ検証済み60枚デッキ

DeepSetsは、候補ごとの48次元表現、合法手集合のmean/max pooling、盤面48次元表現から各候補を再採点します。学習時は同じカードの複製indexを同じsemantic正解集合として扱います。

## 漏洩なし評価

| 指標 | v1 test | v2 test |
|---|---:|---:|
| 非強制semantic exact | 77.13% | **77.48%** |
| 単一選択Top-1 | 77.73% | **78.10%** |
| 単一選択Top-2 | 92.23% | **92.52%** |
| 単一選択Top-3 | **97.30%** | 96.84% |
| 主行動Top-1 | 73.44% | **73.96%** |
| 可変選択数 | 99.63% | 99.63% |

方式・epoch・blend・confidenceはvalidationだけで固定し、その後testを一度評価しました。test後に設定を変更していません。

## 再現

```powershell
.\.venv\Scripts\python.exe scripts\train_lopunny_v2_teacher.py `
  experiments\majkel_lopunny_ml_v1\corpus.npz `
  agents\lopunny\majkel_lopunny_ml_v2 `
  --report experiments\majkel_lopunny_ml_v2\training_report.json

$env:PYTHONPATH = (Resolve-Path vendor).Path
.\.venv\Scripts\python.exe scripts\evaluate_lopunny_top1_runtime.py `
  --agent-dir agents\lopunny\majkel_lopunny_ml_v2 `
  --cache experiments\majkel_lopunny_ml_v1\corpus.npz `
  --split test `
  --report experiments\majkel_lopunny_ml_v2\runtime_test_report.json

.\.venv\Scripts\python.exe scripts\validate_agent.py `
  --agent agents\lopunny\majkel_lopunny_ml_v2

.\.venv\Scripts\python.exe -m pytest -q `
  agents\lopunny\majkel_lopunny_ml_v2
```

## 実行性能

全データ再fit後の既知testログ50試合・3,892 decision再生では、legal 100%、semantic exact 90.44%、平均38.26ms、p95 108.07ms、最大232.90msでした。公式の2秒overage guardより十分小さい範囲です。

ローカルアリーナはv34相手20戦で3-17、crash 0、illegal 0でした。標本が小さく相手も1種類なので勝率比較には使わず、未知盤面を完走できる安全性確認としてのみ扱います。

## 判定

状態は `offline candidate / holdout target not met` です。v1より正直な未知性能は改善しましたが、提出・champion昇格の根拠には不足しています。386試合で試した木8系統のoracle自体が非強制単一選択82.67%だったため、85%へ進むには同一教師・同一デッキの追加ログが最優先です。
