# v33 分析

## 結論

v33では、提案されていた「同一教師・未学習予測によるOOF selector」を実装しました。ただし選択精度は既存研究blendを超えず、不採用です。一方、v31実戦ログで直接測定できたRun Away Draw不足とF/G再構築失敗は、限定ロジックとして本体へ採用しました。

v33は「90%を達成した版」でも「v31を昇格させた版」でもありません。学習上の仮説を漏洩なしで検証し、失敗したselectorを本番から外したうえで、実戦ログ由来の改善だけを安全に試すchallengerです。

## OOF selector

Yushin train 35,704判断をepisode単位で4-foldにしました。各foldを除外して次の6方策を学習し、除外foldへ予測しています。

1. 標準LambdaRank
2. 小葉LambdaRank
3. 大葉LambdaRank
4. 数値ID LambdaRank
5. rank_xendcg
6. recency重み付き大葉LambdaRank

OOFスコアは全train候補行に欠損なく作成しました。メタrankerには180個の盤面・候補特徴と、6モデルの正規化スコア、topとの差、順位、top投票、モデル間統計を与えています。

| 指標 | 結果 |
|---|---:|
| train OOF 6モデルoracle | 84.52% |
| validation 6モデルoracle | 83.25% |
| test 6モデルoracle | 84.62% |
| selector validation | 78.34% |
| selector test | 79.18% |
| 既存v32研究blend validation | 80.21% |
| 採用閾値 | 80.41% |

selectorは採用閾値を1.87ポイント下回りました。原因はselectorだけでなく、今回OOF化できた標準ライブラリ6モデルのoracle自体が、木・DeepSets・Attentionを含む旧8モデルoracle 91.68%より7.06ポイント低いことです。データ量不足は解消しましたが、OOF化したbase familyの多様性が不足しています。

この結果から、「OOFにすれば自動的に90%へ近づく」は反証されました。次にselectorを再開するなら、DeepSets/Attentionをfoldごとに学習するか、方策状態を保持するsequence modelをOOF familyへ含める必要があります。

## ログ由来ロジック

v31のGrimmsnarl対面では、すでに撃てるA〜E状態の処理は上位フィールドと同等でした。差は、Alakazamまたは進化元を探すF/G状態の回復率9.8%対42.6%に集中していました。

同じ60枚デッキなのにDudunsparce使用は1.69回/試合対2.86回/試合で、v31は0回使用の12試合を全敗しています。コードには次の停止条件がありました。

- 次のAlakazamが1ターン以内ならRun Away Drawを使わない
- 相手の残りサイドが1ならRun Away Drawを使わない
- 手札10枚以上でソフト停止
- Dudunsparceがいて手札8枚以上ならEnriching Energyを止める

この前提は「ドローは次アタッカー探索だけのため」というものでした。しかしPowerful Handは手札1枚につき20ダメージなので、手札そのものが攻撃資源です。

v33はこれらの固定停止を外し、同ターン確定KO、盤面崩壊、山札生存だけを停止条件として残しました。また、F/G状態では探索札とDudunsparce cycleをMLより優先させます。

Poffinはcore body 4体で止めず、5体まで展開可能にしました。ShayminはFroslassのダメカンを防げないため、Froslassが見えている盤面では通常の防御札として出しません。BossのFroslass/Munkidori優先は既存v25〜v32にすでにあるため、そのまま継承しています。

## 判断

ローカル同型40戦はv31 20勝、v33 17勝、3分で、v33優位ではありませんでした。Grimmsnarl v7への同条件20戦は両方13勝7敗です。したがってv31をchampionとして維持し、v33は実ラダーで次を測るchallengerとします。

- Run Away Draw回数/ターンが0.309から0.495へ近づくか
- Dudunsparce 0回の試合が減るか
- F/G回復率が9.8%から改善するか
- Powerful Hand 4回以上の試合割合が増えるか
- デッキ切れ、盤面崩壊、確定KO逃しが増えないか

既存Yushin testはv32研究で既に観測済みです。最終的な90%判定には、新しい上位ログから作る完全未閲覧holdoutが引き続き必要です。
