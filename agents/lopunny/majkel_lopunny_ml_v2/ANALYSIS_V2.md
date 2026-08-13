# 設計・分析 V2

## 目標と評価境界

主目標は、最後の50試合を隔離した時系列testにおける `nonforced semantic exact >= 85%` です。全386試合へ再fitしたモデルの既知ログ再生率は合否に使いません。

データ、教師席、submission、60枚deck signature、train 296 / validation 40 / test 50の分割はv1から固定しました。v2の探索ではtest配列を読み込まず、方式をvalidationで固定した後に一度だけtestを評価しました。

## v1残差の診断

v1はtest単一選択Top-3が97.30%である一方、Top-1は77.73%でした。候補発見よりTop-3内の順序が問題です。さらに可変選択数は99.63%なので、個数予測も主要因ではありません。

v1のgraded relevance実験ではturn-setが90%を超えました。これは「このターンに使う候補集合」は認識できるが、「次に使う1手」の順序を解けていないことを示します。

## validation限定の方式比較

| 方式 | validation非強制semantic exact | 判定 |
|---|---:|---|
| v1基準LambdaRank | 75.83% | 基準 |
| 行動種multiclass gate | 76.08% | +0.25pt、単体情報が弱い |
| 高純度canonical policy memory | coverage 0% | 再利用不能 |
| 8 ranker容量・seed・recency sweep 最良単体 | 75.92% | 容量増加だけでは不足 |
| 上位3 ranker z-score ensemble | 76.05% | 改善小 |
| 8 ranker semantic oracle・非強制単一選択 | 82.67% | モデル族上限が85%未満 |
| pairwise Top-8 reranker | 75.95% | +0.12pt |
| DeepSets + ranker blend 選択実験 | 76.23% | 採用 |

policy memoryは、完全・sequence・board・abstractの4 schema、合法手signature、HP量子化、支持数1/2/3/5、純度0.8/0.9/1.0を比較しましたが、別episodeのvalidation MAIN局面を1件も安全に解決できませんでした。相手盤面・手札・ログが異なるためです。

ranker sweepは63/127/255葉、3 seed、最新87.5%/75%episode、均等重みを比較しました。最良平均ensembleでも76.05%、8本のうちどれかが正しいoracleでも主行動79.55%でした。木rankerの選択器を追加しても85%へ届く候補多様性がありません。

## DeepSetsを採用した理由

LambdaRankは `f(state, candidate)` を候補ごとに独立計算します。v1には合法手数の集約特徴がありますが、「この候補AとBが同時にある時だけAを先にする」という比較は間接的です。

DeepSetsは次を計算します。

1. 盤面・手札・ログ・合法手集約を389特徴から48次元へ変換。
2. 各候補114特徴を48次元へ変換。
3. 全合法候補のmean/max poolingを作る。
4. 候補表現、盤面表現、poolingを結合して候補scoreを出す。
5. DeepSets scoreとranker scoreをdecision内z-score化してblendする。

訓練lossは、教師が選んだraw indexだけでなく、同じsemantic actionの複製候補すべてへのsoftmax確率合計を最大化します。モデルはtorchで学習しますが、配布時は75,809 parameterをJSON化し、GELUを含めて標準ライブラリで評価します。

## 最終結果

固定設定をtrainだけでfitした時系列test結果は次の通りです。

- v1非強制semantic exact: 77.13%
- v2非強制semantic exact: 77.48%
- 改善: +0.35ポイント
- 単一選択Top-1: 78.10%
- 主行動Top-1: 73.96%
- 可変選択数: 99.63%
- 85%目標: 未達

DeepSetsはtest MAIN 2,527 decisionへ適用され、validationとtestの両方で小幅改善しました。Top-3は97.30%から96.84%へ低下したため、残差モデルが正解候補をTop-3外へ押す副作用もあります。次版で使うならTop-3を固定したcascadeへ制限する余地があります。

## 85%へ届かない理由

データは386試合です。添付記録のフーディンv34は2,268試合でもstrict Top-1 83.01%でした。本v2では異なる4方式と8 rankerのoracleまで測定しましたが、現特徴量のranker族oracleが82.67%です。

したがって、現在の386試合にselectorを重ねるだけで未知85%を主張する根拠はありません。次の優先順位は以下です。

1. 同一submission・同一deck signatureのログを最低2,000試合へ増やす。
2. 新規ログの最後200試合を固定testとし、今回のtestを開発用へ移す。
3. DeepSets/attentionをepisode OOF化し、Top-3だけを再順位付けする。
4. 複数discardを集合予測ヘッドへ分離する。
5. 教師模倣率と実ラダー結果を別に管理する。

v2は「目標に届かなかった実験」ではなく、木の微調整・暗記・単純selectorでは届かないことを定量化し、異なるlistwiseモデルで未知性能を再現可能に改善した基準版です。
