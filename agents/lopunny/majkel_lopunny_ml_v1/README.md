# majkel_lopunny_ml_v1

現在の1位チーム `Majkel1337`（submission `55137818`）の Mega Lopunny ex / Dudunsparce デッキと選択を模倣する、全 select-context 対応のオフライン候補です。

結論は明確です。386試合すべてでデッキと教師席を検証し、学習済みログのランタイム再生では意味的完全一致 90.21% に達しました。一方、最後の50試合を学習・木数選択から隔離した時系列ホールドアウトでは、強制選択を除く意味的完全一致は 77.13% です。したがって「既知ログの再現9割」は達成、「未知試合の教師選択9割」は未達です。この版は提出用 champion ではなく、データ追加と順序モデル改善のための基準実装です。

## 構成

- `main.py`: 大会エントリポイント。既定は `ml` モード。
- `imitation_features.py`: 状態・候補・行動履歴から作るリーク禁止の特徴量。
- `ranker_model.json`: 各合法候補を順位付けする LambdaRank、900木。
- `count_model.json`: 複数選択数を予測する L1 回帰、200木。
- `tree_runtime.py`: LightGBM JSON を標準ライブラリだけで評価するランタイム。
- `fallback_policy.py`: 比較・アリーナ検証用の任意ルールベース。既定では使わない。
- `deck.csv`: 教師ログで検証した正確な60枚。

候補順位と選択枚数を分離しています。複製カードの位置違いを誤りに数えない semantic exact と、返却indexが完全一致する raw exact の両方を記録します。

## 再現手順

リポジトリ直下で実行します。

```powershell
.\.venv\Scripts\python.exe scripts\build_lopunny_top1_corpus.py `
  --output experiments\majkel_lopunny_ml_v1\corpus.npz `
  --report experiments\majkel_lopunny_ml_v1\corpus_report.json

.\.venv\Scripts\python.exe scripts\train_lopunny_top1_teacher.py `
  experiments\majkel_lopunny_ml_v1\corpus.npz `
  agents\lopunny\majkel_lopunny_ml_v1 `
  --report experiments\majkel_lopunny_ml_v1\training_report.json

.\.venv\Scripts\python.exe scripts\evaluate_lopunny_top1_runtime.py `
  --cache experiments\majkel_lopunny_ml_v1\corpus.npz `
  --split test `
  --report experiments\majkel_lopunny_ml_v1\runtime_test_report.json

.\.venv\Scripts\python.exe scripts\validate_agent.py `
  --agent agents\lopunny\majkel_lopunny_ml_v1

.\.venv\Scripts\python.exe -m pytest -q `
  agents\lopunny\majkel_lopunny_ml_v1
```

学習スクリプトは train だけで候補木を作り、validation の `nonforced_semantic_exact` だけで木数を選び、test は最後に一度だけ評価します。モデル配布時は選ばれた木数で全386試合へ再fitします。そのため、配布モデルを test ログへ再生した90.21%はホールドアウト値ではありません。

## 評価要約

| 評価 | 指標 | 結果 |
|---|---|---:|
| validation 40試合 | 非強制 semantic exact | 75.83% |
| test 50試合 | 非強制 semantic exact | 77.13% |
| test 50試合 | 単一選択 Top-1 / Top-2 / Top-3 | 77.73% / 92.23% / 97.30% |
| test 50試合 | 複数選択数 | 99.63% |
| 全データ再fit後の既知testログ再生 | semantic exact | 90.21% |
| ランタイム再生 | 合法率 / p95 / 最大 | 100% / 60.42ms / 162.03ms |

全データ再fit後の値を未知データ性能として使わないことが、この版で最も重要な運用条件です。詳細は `ANALYSIS_V1.md` と `VALIDATION_REPORT_V1.md` にあります。

## 実行モード

通常は環境変数を設定せずML版を使います。ルール版を比較したい場合だけ次を設定します。

```powershell
$env:LOPUNNY_POLICY_MODE = 'rule'
```

小規模アリーナ20戦ではML版3勝、ルール版5勝でしたが、相手が1種類で標本も少ないため採用判断には使いません。教師模倣という目的に沿って既定値は `ml` のままです。

## 次の合格条件

追加ログ取得後も同じ時系列分割を保ち、隔離した最新testで `nonforced_semantic_exact >= 0.90` を満たすまでは submission-ready に昇格させません。現状はTop-3が97.30%なので、データを増やした後は主行動の再順位付けと、捨て札など集合選択の専用ヘッドを優先します。
