# ml_alakazam 改修最終レポート

## 1. 結論

添付された旧 `ml_alakazam` を基準に、元の設計思想を維持したまま改修した。

維持したものは以下である。

- CABTの合法候補だけを順位付けする imitation learning / ranker
- policy featureへの未来情報・勝敗・相手非公開手札・初期完全デッキの混入防止
- episode単位を基本とするデータ分離
- LightGBMを標準ライブラリだけで実行できるJSON treeへdistillする提出方式
- MLが不確実な場合に既存v12 fallbackへ戻るhybrid設計
- fallbackが確定KOを選んでいる場合にMLで崩さない安全ゲート

最大の原因は、旧 `src/replay_io.py` が正規表現
`/replay/episode_*.json` だけを受理しており、
上位50 ZIPの大半にある `/replays/episode_*.json` を対象外にしていたことだった。

## 2. データ回収結果

| 指標 | 旧版 | 改修版 |
|---|---:|---:|
| Full replay | 164 | **2,058** |
| 複数形 `replays/` から新規回収 | 0 | **1,894** |
| usable trajectory | 162 | **2,074** |
| usable decision | 11,438 | **95,254** |
| legal candidate | 89,729 | **1,097,481** |
| 教師team | 実質1 | **19** |
| submission | 実質1 | **20** |
| deck cluster | 実質1 | **8** |

`rank49 Jack sub54630772` のZIPだけは今回の添付に含まれていない。

## 3. 元プロジェクトを読んで判明した追加問題

### 3.1 seat推定

旧版は、ZIP内manifestにseatがない場合、
「Alakazam deckが片側だけ」という条件を主に使っていた。
Alakazam同士の対戦や同名対戦では誤る可能性がある。

改修版は次の順で判定する。

### 3.2 action alignment

CABT agentは `select.option` のindexを返す。
replayでは観測tに対する同一seatの返答がstep t+1の`action`に格納される。

改修版ではイベントログから行動を推測せず、
この合法option indexを教師ラベルとして直接使用する。
95,255件中95,254件が整列し、未解決は1件だけだった。

### 3.3 split不足

旧datasetではteam/submission/deck splitが
`not_available_single_teacher` になっていた。

改修版では次を独立評価する。

- submission内の後半20%を使うtime holdout
- team holdout
- 最新Majkel提出を使うsubmission holdout
- Majkel完全一致以外の最大clusterを使うdeck holdout

### 3.4 特徴量不足

旧76特徴は、カードIDと盤面の集計は持つ一方、

- 現在の手札打点を行動で壊すか
- KOを維持できるか
- Abra / 進化橋 / Alakazam / Energyのどこが不足しているか
- Hammer対象に特殊Energyがあるか
- Retreat対象の損傷・状態異常
- Energy対象がAlakazamかFezandipitiか
- 低山札でAbilityやTrainerを使う危険

などを浅い木で学びにくかった。

改修版は225特徴とし、上記のaction-state interactionを明示した。

## 4. 教師重み

すべてのフーディン系dataを同一教師として扱わず、以下を使用する。

- team rank
- 勝敗
- Majkel deckからの置換距離
- seat confidence
- action alignment confidence
- rare actionの軽い補正

ただし強い重みは逆効果だった。
ablationではuniformがfull weightingよりtime Top1で約0.38pt高く、
重みの有効性は強く確認できなかった。

そのため重みは平均1へ正規化し、概ね0.65〜1.35に制限した。
敗戦手や遠いdeck variantを一律除外していない。

## 5. Offline評価

| Holdout | Top1 | Top3 | MRR | ECE |
|---|---:|---:|---:|---:|
| Time | **52.36%** | **80.88%** | 0.684 | 0.113 |
| Team | **46.44%** | **79.49%** | 0.647 | 0.146 |
| Submission | **56.67%** | **83.07%** | 0.711 | 0.161 |
| Deck | **50.57%** | **79.73%** | 0.670 | 0.111 |

これは旧レポートの74.17%より低く見えるが、評価条件が異なる。

旧値は実質単一のrank1 Majkel deck内のtime splitであり、
semantic labelも用いていた。
改修版は、別team・別submission・別deckへ一般化する
より厳しいexact legal-option評価である。

初回拡張モデルから文脈特徴を追加した結果、

- Time Top1: +7.17pt
- Team Top1: +9.67pt
- Submission Top1: +1.81pt
- Deck Top1: +6.97pt

改善した。

最新版Majkelの将来sliceでは、singular-only学習61.60%に対し、
全team拡張学習は66.69%で、+5.09ptだった。

## 6. action type別の判断

Time holdoutの主要結果は以下である。

| Action | Top1 | 判断 |
|---|---:|---|
| Ability | 88.74% | ML利用可 |
| Attack | 61.55% | confidence gate付き |
| Bench | 63.16% | confidence gate付き |
| Evolve | 61.91% | confidence gate付き |
| Energy | 35.89% | **0.85以上のみML** |
| Hammer | 41.30% | **常にfallback** |
| Xerosic | 22.27% | **常にfallback** |
| Boss | 4.93% | **常にfallback** |
| Retreat | 4.15% | **常にfallback** |

focus actionを強く重み付けした専用モデルでは、
HammerやXerosicは改善したが、全体Top1が最大約7pt低下した。
単一モデルとして採用するのは危険と判断した。

将来はgeneral rankerとaction別expertを分離した
mixture-of-expertsが有力である。

## 7. Hybrid runtimeの安全設計

改修した提出runtimeは次の制約を持つ。

- `ACTIVE MAIN`かつ1枚選択だけをML対象にする
- nested search、target選択、複数選択はfallback
- Boss / Retreat / Xerosic / Hammerは必ずfallback
- Energyはprobability 0.85以上かつmargin 0.12以上だけML
- fallbackが確定KOを選んでいる場合は上書きしない
- 最後の盤面を消すDudunsparce abilityと低山札使用を防止
- MLは渡された合法候補index以外を返せない
- model読込失敗、timeout、推論例外はfallback
- runtime依存はPython標準ライブラリと公式`cg`だけ

## 8. JSON tree runtimeの修正

元のdistillerはLightGBMの数値splitしか正しく扱っていなかった。

今回のrankerはcard IDやaction typeをcategorical featureとして使用するため、
LightGBM treeには `decision_type == "=="` と
`"0||1||2"` のようなカテゴリ集合thresholdが現れる。

旧distillerはこれをfloatへ変換しようとするため提出model生成に失敗する。
改修版はカテゴリ集合をJSONへ保存し、
runtimeで整数membership判定する`lightgbm_tree_v2`に変更した。

500候補行でLightGBM native predictionとdistilled runtimeを比較し、
最大絶対誤差は**0.0**だった。

## 9. テスト・検証

- replay/replays両構造のsynthetic test
- 2,074 trajectoryのseat/deck監査
- 95,254 decisionの一意性
- team/submission/deck/time splitの存在確認
- policy feature leakage denylist
- opponent private hand ID不変性
- native LightGBMとcategorical JSON runtimeの完全一致
- model失敗・timeout・nested selection fallback
- focus action hard fallback
- Energy high-confidence gate
- 既存v12 golden-state suite
- deck 60枚、ACE SPEC、fallbackロジック

実際のKaggle Rating改善は、公式engineと対戦poolがこの添付環境にないため確認していない。
旧battle結果を新modelの結果として流用せず、stale artifactとして分離した。

## 10. 残る危険

1. rank49 Jack dataが未取得
2. Boss / Retreat / Xerosicは依然としてML品質が低い
3. Hammer expertはtimeでは改善したがsubmission一般化が弱い
4. Teacher weightingの優位性は未確定
5. Offline Top1改善が実戦勝率改善を保証しない
6. 現在のmodelはMAIN actionだけを学び、target選択はfallback依存
7. 公式engineでのbattle smokeとValidation Episodeが未実施

## 11. 推奨する次段階

1. この版を旧ML版ではなく、v12 fallbackとのA/B対象にする
2. 同一対面・席順交換で最低100〜200戦
3. ML override率、fallback率、違法手、初攻撃、攻撃継続、山札切れを記録
4. Boss/Hammer/Xerosic/Retreatはaction別expertを別modelとして学習
5. expertを導入する場合も、general modelとのgatingをholdoutで検証
6. rank49 Jackを追加してmanifest・holdoutを再生成
