# 検証レポート V2

実施日: 2026-08-02

## データ

- 教師: `Majkel1337`
- submission: `55137818`
- manifest: 396行
- 正常・同一デッキreplay: 386
- 取得エラー: 10
- deck mismatch: 0
- 分割: train 296 / validation 40 / test 50 episode

## 方式選択

行動種gate、canonical memory、8 ranker sweep/ensemble/oracle、pairwise reranker、DeepSetsをvalidationだけで比較しました。最終方式は900木ranker、11epoch DeepSets、`deep_z + 2 * base_z`、confidence 0.20、200木count headです。

## 正直な時系列holdout

| 指標 | validation | test |
|---|---:|---:|
| v2非強制semantic exact | 76.01% | 77.48% |
| v2非強制raw exact | 73.44% | 73.68% |
| v2単一選択Top-1 | 76.42% | 78.10% |
| v2単一選択Top-2 | 91.38% | 92.52% |
| v2単一選択Top-3 | 96.26% | 96.84% |
| v2主行動Top-1 | 72.33% | 73.96% |
| 可変選択数 | 100.00% | 99.63% |

test非強制semantic exactの目標85%は未達です。v1 test 77.13%に対して+0.35ポイントです。

## 配布ランタイム再生

全386試合へ固定サイズで再fitした配布モデルを、既知testログ50試合・3,892 decisionへ再生しました。

- legal rate: 100%
- semantic exact: 90.44%
- raw exact: 88.26%
- count accuracy: 99.05%
- mean: 38.26ms
- median: 29.99ms
- p95: 108.07ms
- max: 232.90ms
- ranker/deepset/count model load errors: 0

これは `deployment-refit runtime parity/resubstitution` で、未知性能には数えません。

## 静的・パッケージ検証

- Python compile: passed
- agent tests: 6 passed
- exact 60-card deck test: passed
- 900木ranker / 200木count / DeepSets JSON load: passed
- forced selection / zero selection: passed
- fake official cgを使ったsubmission bundle: passed
- bundleに3モデル、特徴量、runtime、policy baseを同梱: passed
- README/metadata/tests/training reportをbundleから除外: passed

## ローカルアリーナ

`alakazam_ml_v34` 相手20戦、seed 5515:

- 戦績: 3-17
- crash: 0
- illegal select: 0
- seat pool平均: 48.66ms / 52.87ms per move

勝率は標本が小さく相手が1種類なので比較根拠にしません。未知盤面の完走・合法性・時間だけを安全性証拠とします。

## 判定

`offline candidate / holdout target not met`

v1からの未知模倣性能改善は再現しましたが、85%と対戦強度の両ゲートを満たしていないためsubmission-ready、championには昇格しません。
