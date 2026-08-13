# Grimmsnarl ML v19

## データ

- 既存current-top4: 1,014 relation
- 新規1000+教師: 600 relation（各200戦）
- 重複差替え: 350 relation
- 統合後: 1,264 relation / 1,238 episode
- 95,664 decision / 466,463 candidate row
- 新規教師team: 16422241、16452116、16561259
- 公開score: 1113.7、1151.0、1116.3

分割は教師ごとの時系列分割で、train 72,812、validation 11,432、test 11,420 decision。テスト期間は学習に使用していない。

## 保存ログ検証

新規3教師の各最新24 test episode、合計72戦・2,708 MAIN判断を実ランタイムでteacher forcingした。

| Agent family | correct | Top-1 |
| --- | ---: | ---: |
| v8系 | 1,839 | 67.91% |
| v9系 | 1,963 | 72.49% |
| v19勝利4倍 | 1,927 | 71.16% |

v19はv8系を上回るが、v9系よりstrict imitationは1.33ポイント低い。この差は未解決の不確実性として扱う。

## 変更範囲

- `ranker_model.json`: 新しい勝利4倍575-tree model
- `ml_runtime.py`: teacher-unconditioned modelを読み、古いv6 teacher escalationを停止
- `main.py`: 公開routeを残存するv18 safety layerへ通知
- `metadata.json`: 学習・候補棄却・検証結果を記録

`deck.csv`、`ml_features.py`、`ml_planner.py`、`ml_residual.py`、`attack_access.py`、`wall_break.py`、`mirror_prize.py`等はv18と同一。phase expertや`ranker_model_v9.json`は最終版に含めない。

## 限界
