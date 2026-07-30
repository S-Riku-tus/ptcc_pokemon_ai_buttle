# alakazam_ml_v33

v33は、v31を実戦champion、v32をオフラインchallengerとして保存したまま作成した、ログ根拠のロジック改善版challengerです。末尾の「v32を作成」は全体の文脈からv33作成の指定として扱っています。

学習面では、Yushin Itoのtrain 35,704判断をepisode単位で4分割し、6種類のLightGBM方策から完全OOFスコアを生成しました。候補ごとのモデルスコア、順位、margin、投票、180盤面特徴を合わせた214特徴のメタrankerを学習しています。

OOF selectorはvalidation 78.34%、test 79.18%でした。採用条件は、凍結済みv32研究blendのvalidation 80.21%を最低0.20ポイント上回る80.41%です。条件を満たさなかったため、selectorは`enabled: false`のshadow資産として保存し、提出時には6個の大きなbase artifactを自動除外します。testはvalidationで構成と採否を確定した後に1回だけ評価しました。

実戦ランタイムはv32のrecency LambdaRank、Majkel盤面メモリ、安全ガードを継承し、v31対戦ログで直接測れた弱点だけを限定的に上書きします。

- Run Away Drawの`backup_eta <= 1`と「相手残りサイド1」の停止条件を削除
- 固定手札10枚停止を廃止し、同ターンの確定Powerful Hand KOだけを手札側の停止条件に変更
- Dudunsparce＋手札8枚でEnriching Energyを止める条件を削除
- F/G再構築状態ではHilda、Dawn、Poké Pad、Run Away Draw、進化元展開を優先
- Buddy-Buddy Poffinを4 core bodyの時点でも使い、Abra/Dunsparceの幅を確保
- Froslassがいる盤面ではShayminを防御札として数えない
- 盤面崩壊と山札切れの安全ガードは維持

検証は185テスト成功、静的agent検証成功、OOF compact-runtime 25盤面の最大スコア誤差0.0です。v31との同seed・先後交替40戦はv31 20勝、v33 17勝、3分でした。Grimmsnarl v7への各20戦はv31、v33とも13勝7敗です。クラッシュ・違法手は0でした。

したがって、v31はchampionのままです。v33はラダーでRun Away Draw使用率、F/G回復率、Powerful Hand回数を測るchallengerであり、現時点で昇格版ではありません。詳細は`ANALYSIS_V33.md`と`VALIDATION_REPORT_V33.md`を参照してください。
