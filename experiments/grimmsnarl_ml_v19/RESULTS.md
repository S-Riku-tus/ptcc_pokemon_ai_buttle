# Grimmsnarl ML v19

## 結論

v19は、v18の安全層と同じ60枚を維持し、根幹のrankerを再学習したchallengerである。
最終版はpolicy stitchingを行わない単一モデルで、上位1000+の3教師から取得した新しい600戦を既存current-top4 corpusへ統合し、教師が勝った対局の判断を4倍に重み付けした。

ラダー検証はユーザー指定により行っていない。ローカルarena 200戦ではv8、v9、v15、v18の代表4系統に合計114-86、クラッシュ0、不正選択0だった。

## データ

- 既存current-top4: 1,014 relation
- 新規1000+教師: 600 relation（各200戦）
- 重複差替え: 350 relation
- 統合後: 1,264 relation / 1,238 episode
- 95,664 decision / 466,463 candidate row
- 新規教師team: 16422241、16452116、16561259
- 公開score: 1113.7、1151.0、1116.3

分割は教師ごとの時系列分割で、train 72,812、validation 11,432、test 11,420 decision。テスト期間は学習に使用していない。

## 候補選定

当初の対面別conditioned modelと、v19序盤・v9中終盤を使う二段modelを検証した。しかし二段版はローカルv9戦で6-14となり、元分析が警告したpolicy stitchingを再発させたため完全に撤去した。

単一モデルの比較は次の通り。

| 候補 | 時系列test Top-1 | test MAIN | v9ローカル初回20戦 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 未重み付け | 80.44% | 73.64% | 11-9 | 比較用 |
| 勝利2倍 | 80.18% | 74.01% | 8-12 | 不採用 |
| 勝利4倍 | 80.06% | 73.14% | 12-8 | 採用 |

勝利4倍はstrict imitationを少し失う代わりに、勝った教師系列の方策を強く学ぶ。最終モデルは575木、822特徴量、teacher IDなし。勝敗ラベルは学習時の重みにだけ使い、推論入力には含めない。

## 保存ログ検証

新規3教師の各最新24 test episode、合計72戦・2,708 MAIN判断を実ランタイムでteacher forcingした。

| Agent family | correct | Top-1 |
| --- | ---: | ---: |
| v8系 | 1,839 | 67.91% |
| v9系 | 1,963 | 72.49% |
| v19勝利4倍 | 1,927 | 71.16% |

v19はv8系を上回るが、v9系よりstrict imitationは1.33ポイント低い。この差を隠さず、ローカル勝敗とのトレードオフとして扱う。

## ローカルarena

native engine shuffleはPython seedで完全に固定されず、対戦は非ペアである。したがって因果的な勝率証明ではなく、候補選別・クラッシュ・不正選択・極端な退行の検査として使った。

| 相手 | 戦数 | v19成績 |
| --- | ---: | ---: |
| v8 | 20 | 11-9 |
| v9 | 100 | 55-45 |
| v15 | 60 | 36-24 |
| v18 | 20 | 12-8 |
| 合計 | 200 | 114-86 |

全200戦でクラッシュ0、不正選択0。最終フォルダの追加smokeはv9に8-12、v15に12-8で、上記集計に含む。

## 変更範囲

- `ranker_model.json`: 新しい勝利4倍575-tree model
- `ml_runtime.py`: teacher-unconditioned modelを読み、古いv6 teacher escalationを停止
- `main.py`: 公開routeを残存するv18 safety layerへ通知
- `metadata.json`: 学習・候補棄却・検証結果を記録

`deck.csv`、`ml_features.py`、`ml_planner.py`、`ml_residual.py`、`attack_access.py`、`wall_break.py`、`mirror_prize.py`等はv18と同一。phase expertや`ranker_model_v9.json`は最終版に含めない。

## 限界

ラダーを実施していないため、将来ratingや1000+帯でのdriftは保証できない。v19は、保存ログ・時系列test・200戦local arenaに基づく、現在最も根拠の強いchallengerである。
