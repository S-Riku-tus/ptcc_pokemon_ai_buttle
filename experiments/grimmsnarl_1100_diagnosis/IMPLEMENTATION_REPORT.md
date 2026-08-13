# Grimmsnarl 1100 診断・評価基盤 実施報告

実施日: 2026-08-13 JST

## 結論

Stage 0（再現可能なオフライン比較器）と Stage 1（施策サイズの事前ゲート）は実装・検証まで完了した。

現時点では v21 を v20 より強い版とは認定しない。主要3対面代理を使った300 seedブロック、1,200 primary gamesでは、v21-v20 は **+1.33勝率ポイント、95% CI [-2.83, +5.50]、paired exact p=.568** だった。v21は十分大きく行動を変えるが、改善と悪化が相殺されている。

また v20 は v8 より強いとも認定できない。100ブロックでは v20-v8 が **-2.50pt、95% CI [-9.38, +4.38]、p=.609** だった。したがって、全履歴のchampionを確定するまでの暫定回帰基準は v8 とし、v20/v21は条件付きで有用な部品候補として残す。

1100はデッキ変更なしでも到達可能である。同一60枚・同一デッキhash `9714ab5c3996f6cc` が55チームで使われ、1220.2、1158.9、1147.4、1142.4、1135.8に到達している。まず方策差を取りに行くべきで、デッキ変更が最優先という証拠はない。

## 今回実装したもの

### 1. 検証済み native RNG 制御

`scripts/cg_seed.py` を追加した。

- `vendor/cg/cg.dll` だけを対象にする。
- SHA-256 `e758cdb...ee17f9` と対象命令バイトの両方を照合し、未知buildではfail closedする。
- DLLファイルは変更せず、プロセス内の命令だけを一時的にseed経路へ切り替える。
- シャッフル、コイントス、サイド配置を含むengine乱数ストリームを試合ごとにresetする。
- 終了時に元命令へ戻す。

実agentの v21-v8 で同seedを再実行し、可視観測、全行動列、結果、手数が一致した。異なるseedでは異なる軌跡になった。nativeのopaque `search_begin_input` 生バイトはプロセス依存で一致しないが、agent行動と可視ゲーム軌跡への影響がないことを実測している。各比較runでも重複試合校正を必須にした。

### 2. common-random-number ペアガントレット

`scripts/paired_gauntlet.py` と `configs/paired_gauntlet/grimmsnarl.json` を追加した。

各 `(opponent, seed, first/second)` について、championとchallengerへ同じ乱数ストリームを与える。両方の先後を走らせ、seed単位でcluster化したCIを出す。

出力する主指標は以下。

- challenger-champion のペア勝率差
- seed-cluster 95% CI と近似80% MDE
- challengerだけ勝ち / championだけ勝ち / 同結果
- discordant pairだけを使うexact McNemar/sign p値
- 対面別、実際の先攻/後攻別の同じ指標
- raw勝率とWilson CI
- error、重複校正不一致、throughput
- 各ゲームのseed、座席、可視観測hash、行動hash、結果

長いrunには5%刻みの進捗表示も追加した。worker初期化失敗はworker再生成ループにせず、runをinvalidにする。

### 3. 施策サイズの事前ゲート

`scripts/policy_impact_gate.py` を追加した。

保存局面のteacher-forced sweep/footprintから、1試合あたりの変更行動数を判定する。

| 変更行動数/試合 | 判定 |
|---:|---|
| <0.5 | 実装・対戦評価を打ち切る |
| 0.5〜2.0 | 2,000 paired games級で判定 |
| >2.0 | 大きい候補として実装可能 |

これは露出量のゲートであり、改善の証明ではない。

### 4. 上位パイロット条件付き代理

既存v8 rankerを、学習済みの1220.2-rated pilotのteacher code `0` で全判断に条件付ける `grimmsnarl_ml_v8@teacher_code=0` を評価器で扱えるようにした。agent/modelファイル自体は変更しない。

この代理の保存データtop-1 fidelityは.797であり、上位提出物そのものではない。そのため従来教師proxyと半々にし、結果を必ず別々に表示する。

## 再現性と性能

### 同一バイナリ自己比較

v21-v21を1,000 primary gamesで比較した。

- valid: true
- complete pairs: 500
- challenger only / champion only: 0 / 0
- ペア差: 0.000
- calibration: 4/4一致
- error: 0
- wall time: 7分38秒
- throughput: 7,892 games/hour

相手に使ったgeneric Crustleにはv21が499/500勝った。このrunは再現性検査としては有効だが、強さ評価には相手が弱すぎる。この結果を受け、generic Kangaskhan/Crustleは正式ガントレットから除外した。

### 別プロセスでの完全再現

v20-v21の同一100ブロックrunを、先攻記録機能の追加前後に別プロセスで実行した。校正を含む412ゲームをキーごとに比較し、次の全項目が **0不一致** だった。

- result
- evaluated win
- moves
- canonical observable hash
- action hash
- error

## v20 と v21 の正式比較

評価相手は次の3層。v21のラダー59戦分布のうち、信頼できる代理を用意できた58%だけを表す。

- Grimmsnarl 1220-pilot conditional proxy: 重み.23
- Grimmsnarl incumbent-teacher control: 重み.23
- Alakazam v35: 重み.12

未表現は Kangaskhan/Crustle 12%、Mega Lucario、Dragapult、Ogerpon、long tailである。したがって以下は「主要対面での相対比較」であって、ラダー全体の無偏推定ではない。

300ブロックの結果:

| 層 | ペア数 | v20 WR | v21 WR | v21-v20 | seed-cluster 95% CI | exact p |
|---|---:|---:|---:|---:|---:|---:|
| 合成 | 600 | .4850 | .4983 | +.0133 | [-.0283, +.0550] | .568 |
| 1220代理 | 238 | .4580 | .5126 | +.0546 | [-.0103, +.1195] | .117 |
| 従来教師 | 238 | .4454 | .4160 | -.0294 | [-.0989, +.0401] | .457 |
| Alakazam | 124 | .6129 | .6290 | +.0161 | [-.0681, +.1004] | .845 |

全体のdiscordant pairは v21だけ勝ち79、v20だけ勝ち71、同結果450。v21は勝敗を25%のペアで動かしたが、正負がほぼ相殺した。

独立した100ブロックrunでは全体差が厳密に0で、実先後別は以下だった。

| 手番 | ペア数 | v20 WR | v21 WR | v21-v20 | 95% CI |
|---|---:|---:|---:|---:|---:|
| 先攻 | 100 | .610 | .640 | +.030 | [-.051, +.111] |
| 後攻 | 100 | .470 | .440 | -.030 | [-.111, +.051] |

どちらも未確定だが、v21を「後攻改善版」とする証拠はなく、観測方向は逆である。

判定: **v21をv20より強い全体版としては採用しない。** 1220代理での正の部分は、条件付き方策候補として保持する。

## v8 と v20 の比較

100ブロック、400 primary games:

| 層 | ペア数 | v8 WR | v20 WR | v20-v8 | 95% CI | exact p |
|---|---:|---:|---:|---:|---:|---:|
| 合成 | 200 | .565 | .540 | -.025 | [-.0938, +.0438] | .609 |
| 1220代理 | 78 | .5513 | .4744 | -.0769 | [-.2094, +.0555] | .327 |
| 従来教師 | 80 | .5000 | .5500 | +.0500 | [-.0415, +.1415] | .541 |
| Alakazam | 42 | .7143 | .6429 | -.0714 | [-.1940, +.0512] | .549 |

先後別では v20-v8 が先攻+1pt、後攻-6ptだった。ともに不確実だが、v20をv8より上の回帰基準にする根拠はない。

判定: **v8を暫定回帰基準として維持する。** ただしv15/v19代表をまだ同じ装置で比較していないため、全履歴champion確定とはしない。

## 既存診断の修正

### 正しかった点

- 40〜60戦の単発ラダーでは+5pt級の差を判定できない。
- 同一v19 binaryが978.3と904.7になり、rating単体は版間採用指標にできない。
- wall-break単体は71/529=.134手/試合で小さすぎる。
- 後攻と最初2ターンのsetupは大きな観測軸である。
- ラダーの相手分布と取得時期を統制する必要がある。

### 強すぎた、または誤っていた点

1. 「v21は1試合.13手しか変えない」はwall-break単体の値である。v20-v21全体では133試合、11,658判断中774判断、**5.82手/試合、97.7%の試合**で行動が変わる。v21全体は測定可能な大きさである。
2. 「v4以降17版は同じ母集団」はラダーだけからは区別不能という意味なら正しいが、方策が同じという意味では誤り。CRNでv20-v21は25%のペア勝敗を変えた。
3. 「教師平均ratingが模倣の厳密な上限」は成立しない。ただし平均模倣が上位固有の行動を薄める構造リスクはある。
4. 敗戦中のmiss発生率は救済可能勝率ではない。重複・交絡があるため、必ず介入再戦で測る。
5. rating-noise診断の最終rating simulationは仮定依存が強く、`identical_pair_95pct_gap=578`などを校正値として採用しない。直接確認できた同一binary約74点差だけを確定事実とする。

## 1100が可能と判断する根拠と限界

同一60枚の上位5パイロットは概ね55〜58%の実測勝率で1135〜1220に到達している。同一デッキelite対lowの対面標準化勝率差は **7.42pt、95% CI [4.33, 10.73]** だった。デッキよりパイロット方策に大きな改善余地がある。

ただし `rating points / win-rate point` の換算から「あと10.87pt必要」と断定しない。ratingは相手強度、時間、更新過程に依存し、線形・因果的ではない。

デッキにはOgerponという構造弱点がある。上位5でもOgerpon勝率は概ね15〜31%である。最新メタでOgerpon比率が上がる場合、方策だけで1100を安定させられない可能性がある。

さらに上位コーパスは主に8月5日まで、`top100_current`も8月7日までで、v21の8月12〜13日より約6〜8日古い。top100は取得時点でmedian 13 games、56/100が20戦未満だった。最新メタを再取得するまで、対面比率を固定的な真値として扱わない。

## 次に作るagentの仕様

新しい小ruleを足すのではなく、次の順序で進める。

### P0. retrospective champion selection

v1〜v21を保存局面の行動距離でcluster化する。ほぼ同じ版は代表1つにまとめ、まず v8、v15、v19、v20、v21を50〜100ブロックでscreenする。上位2代表だけを300ブロック以上で再比較する。

採用条件:

- 全体差の95% CI下限 > 0、または
- 実用非劣性を確認したうえで主要な弱点層に+5pt以上
- error 0、duplicate calibration不一致0

### P1. setup funnel の計測と直接最適化

初期配置から勝敗までを次のfunnelで保存する。

1. 初期 Impidimp 数・active/bench配置
2. 自ターン2までのSpikemuth
3. 自ターン2までのGrimmsnarl ex
4. 自ターン2までのready/attack
5. first prize取得ターン
6. 終局勝敗

既存の後攻観測値:

- turn2 energy in play: v21 2.304、top5 3.187
- turn2 Grimmsnarl ex on board: v21 .348、top5 .489
- 初期 total Impidimp: v21 .576、top5 .684

候補の中間ゲートは、後攻turn2 energy +0.5、Grimmsnarl到達+10pt、初期Impidimpを少なくとも.65。これらは診断用で、最終採用はCRN勝敗差で行う。

### P2. discordant-pair value learning

v20だけ勝った71ペア、v21だけ勝った79ペアなどをseed単位で再生し、最初に方策が分岐した合法選択を記録する。同じ初期乱数下で最終勝敗が逆転した方向を価値信号にする。

- 全ログの平均模倣より、勝敗に結び付く信号を優先する。
- 相手名、終局後情報、将来のカードなどを特徴へ混ぜない。
- first-divergence以前に観測可能な特徴だけで学習する。
- off-policy supportが低い局面は学習せず、v8へfallbackする。

### P3. 条件付きroutingを事前検証

v21は1220代理で正、従来教師proxyで負だった。v8/v20でも同じ異質性が出た。

相手の自ターン1〜2までの公開カード・active/bench・stadium・energyだけで、方策タイプを識別できるかをcross-validationする。識別不能ならrouting案を棄却し、両proxyで同符号になるsetup改善だけを探索する。

### P4. 対面代理を拡張

現在の正式比較は分布の58%しか表現しない。次の順に上位ログから専用代理を作る。

1. Kangaskhan/Crustle（12%）
2. Mega Lucario（7%）
3. Dragapult（7%）
4. Ogerpon / current long tail

generic policyはproxy強度ゲートを通さない。候補agentの勝率が95%以上になる相手は再現性self-check専用に降格する。

## 今後使う指標

### 採用を決めるprimary metrics

- CRN paired win-rate effectとseed-cluster CI
- discordant pairの方向・exact p
- matchup × first/second interaction
- proxy coverageとproxy fidelity
- calibration mismatch / illegal action / timeout
- effect per compute-hour

### 原因を絞るsecondary metrics

- setup funnel conversion
- turn2 energy、ready attacker、hand、deck runway
- first prize turn、prize tempo、overkill/wasted damage
- game-winning attack/evolve/retreatのirreversible regret
- low-deck drawとdeckout risk
- teacher別agreement、pilot間disagreement
- candidateが保存データsupport内にいる割合、effective sample size
- first-divergence時点の価値差

### 採用に使わない単独指標

- 1回の最終rating
- test AUC / MAIN top-1の小数点以下
- 中間盤面指標だけの改善
- 敗戦ログでのrule発生率
- 多重比較補正なしの局所p<.05

## ラダー運用

1. offline championを固定する。
2. challengerはimpact gateと300-block CRN gateを通過させる。
3. champion/challengerを同時刻に提出する。
4. 最低150戦をpoolする。ただし150戦は+5ptの確定には不足し得るため、最終的なdistribution-shift確認と位置付ける。
5. ratingではなく、対面・先後・相手ratingで層別した勝率差を見る。
6. 提出直後からepisodeを保存する。

## 再実行

```powershell
# engine/agent再現性
.\.venv\Scripts\python.exe scripts\probe_cg_seed.py `
  --agent-a grimmsnarl_ml_v21 --agent-b grimmsnarl_ml_v8 `
  --repeats 2 --distinct-seeds 2 `
  --report artifacts\seed_probe.json

# v20-v21の保存局面上の影響量
.\.venv\Scripts\python.exe scripts\policy_impact_gate.py `
  experiments\grimmsnarl_ml_v21\footprint_v20_vs_v21.json `
  --report artifacts\v21_impact.json

# 正式比較
.\.venv\Scripts\python.exe scripts\paired_gauntlet.py `
  --config configs\paired_gauntlet\grimmsnarl.json `
  --blocks 300 `
  --out artifacts\paired_gauntlet\candidate.json
```

## 検証

- relevant pytest: 44 passed
- Python compileall: passed
- seed probe v21-v8: approved for CRN
- 1,000-game identical-policy self-check: valid
- 300-block v20-v21: valid, error 0, calibration 12/12
- separate-process 100-block reproduction: 412/412 game rows fully matched

