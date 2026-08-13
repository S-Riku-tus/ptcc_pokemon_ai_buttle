# v29 validation report

## Teacher imitation

### Held-out episodes

- test: 2,463 MAIN decisions
- semantic top-1: 66.30%
- semantic top-2: 83.56%
- semantic top-3: 90.95%
- v28 fallback semantic: 48.72%
- v28/v29 oracle: 74.18%

### Frozen full corpus

- 30,359 decisions、runtime error 0
- v28 semantic: 59.57%
- v29 semantic: 84.82%
- v28 MAIN semantic: 41.12%
- v29 MAIN semantic: 82.29%

full corpusは学習episodeを含むため、held-out指標とは分けて扱います。

## Reliability checks

- v29 test suite: 169 passed、0 failed
- agent validation: pass
- agent validation warnings: 0
- deck: 60 cards
- model features: 422
- model trees: 278
- model artifact: 1,799,898 bytes
- ranker SHA-256: `9724331c95ee88abd98b3531af862fa4fea2c399d1c1eadca8a7ef19e6811db4`

リポジトリ全体のtestにはv29と無関係な既存13失敗があります。主因は欠落した旧agent directory、未配置のSpidops reconstruction、既存v10 assertionです。v29 directory単独では全testが通っています。

最終的な昇格判定には、v29の実提出と対戦ログ取得が必要です。
