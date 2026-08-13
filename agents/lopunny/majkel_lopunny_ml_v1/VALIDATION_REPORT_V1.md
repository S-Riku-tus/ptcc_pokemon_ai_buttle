# 検証レポート V1

実施日: 2026-08-02

## データ完全性

- manifest: 396行
- 正常replay: 386
- 取得エラー: 10
- 教師名: `Majkel1337`
- submission: `55137818`
- 教師席: manifestと`TeamNames`で試合ごとに解決
- 同一60枚を確認: 386/386
- deck mismatch: 0

## 漏洩なし時系列評価

train 296試合だけでfitし、validation 40試合だけで木数を選び、最後のtest 50試合は選択後に一度だけ評価しました。

| 指標 | validation | test |
|---|---:|---:|
| 非強制semantic exact | 75.83% | 77.13% |
| 非強制raw exact | 73.26% | 73.33% |
| 単一選択Top-1 | 76.22% | 77.73% |
| 単一選択Top-2 | 91.28% | 92.23% |
| 単一選択Top-3 | 95.93% | 97.30% |
| 主行動Top-1 | 72.05% | 73.44% |
| 可変選択数 | 100.00% | 99.63% |

合格基準 `test nonforced semantic exact >= 90%` は未達です。

## 配布ランタイム整合性

選択した木数で全386試合に再fitした配布モデルを、既知のtestログ50試合・3,892意思決定へ再生しました。

- legal rate: 100%
- semantic exact: 90.21%
- raw exact: 88.08%
- count accuracy: 99.05%
- mean: 22.32ms
- median: 17.32ms
- p95: 60.42ms
- max: 162.03ms

この90.21%は `deployment-refit runtime parity/resubstitution` です。testログ自体が最終再fitに含まれるため、ホールドアウト根拠ではありません。

## 静的・単体検証

- agent validator: 60枚、18種、警告0
- runtime tests: 5 passed
- Python compile: passed
- 実モデルload: passed
- 強制候補選択: passed
- selection数0: passed
- デッキ返却のcopy/reset: passed

## 判定

`offline candidate / holdout target not met`

提出・champion昇格は保留します。次回は追加ログを同一のデッキ・教師席検査へ通し、最新時系列testで90%を超えた場合に再判定します。
