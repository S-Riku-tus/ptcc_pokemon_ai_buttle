# Grimmsnarl ML v28 — design and validation

Date: 2026-08-15  
Agent: `agents/grimmsnarl/grimmsnarl_ml_v28`  
Artifact: `artifacts/grimmsnarl_ml_v28_submission.tar.gz`

## 結論

v28 は、v27へ探索を追加する延長ではない。通常の賞品レースを v25 の
AlphaTCG-conditioned ranker に任せ、公開情報からダメージ無効壁を確認した
時だけ、履歴を同期済みの v22 ranker に切り替える mixture policy である。

これは「強そうな処理を増やす」変更ではなく、次の観測を一つの方策に統合する。

| cell | v22 | v25 | v28 owner |
|---|---:|---:|---|
| ordinary race | 94-50 = 0.653 | 25-9 = 0.735 | v25 |
| wall/tank | 27-23 = 0.540 | 3-15 = 0.167 | v22 + guards |

この点推定を v25 の露出比率に当てると、v22 約0.614に対して mixture は
約0.667、Elo換算で約+40となる。ただし各cellは小標本であり、これは事前予測で
あって検証済みの改善量ではない。

安定+200 Eloを達成したとは主張しない。既存推定では真の実力は約970、top 50
に+78、top 10に+192が必要で、既知のマッチアップ穴を全部0.55へ直しても
約+25しか説明できない。v28は、現時点のデータから正当化できる最も大きく、
かつ測定可能な challenger である。

## なぜこの構成か

### 1. v27の変更はラダーで測られていなかった

v27は35試合2,755単一選択中、v22と異なる決定が8個だけだった。belief searchは
301回検討、23回探索、336 branchを評価して最終override 0。したがって853.3は
探索やguardの強さを測った数字ではない。

v28は同じ35試合分布で v22 ranker から411決定（14.9%）変更する。事前登録した
「35試合50決定以上」のfootprint gateを8倍以上で通過する。

### 2. 通常盤面では現行パイロットのほうが良いproxyだった

AlphaTCGの時系列holdout 14試合901決定で、strict Top-1は v22 77.14%に対し
v25 86.35%（+9.21pp、episode bootstrap 95% CI +7.23〜+11.23）。shipped
runtimeのMAINでは68.57%対82.04%だった。v25 ladderの通常raceも25-9だった。

v28は通常、ミラー、Teal Mask Ogerponを v25へ送る。rankerには
`first_player_is_self` が含まれるため、後攻固有の現在パイロットの選択も学習
対象である。v27ログ34試合と対応させると、v22からの変更は先攻10.18/試合、
後攻13.12/試合で、崩壊していた後攻側に十分なfootprintがある。

### 3. v25の弱点は壁cellへ集中していた

v25は壁/tankで3-15、v22は27-23。v25は壁対面でFroslass同時展開が
0.30/turn（v22 0.89）、無効攻撃が増え、deck-outも発生した。そこでv28は
Crustle、Cornerstone型、Neutralization Zoneが公開された時点でv22へ切り替え、
さらにv26で作ったwall trajectory、wall breaker、deck clockを適用する。

両rankerは全決定を採点し、最終着手を両方へcommitする。壁が途中で判明しても、
切替先v22のturn-historyは実際のゲームと一致する。

### 4. context 22 / context 5は未評価ではなく、既にelite ruleだった

multi-pickはrankerを通らないが、v22 fallbackは3,710 exact-deck上位replayから
作られている。

| decision | candidate | elite |
|---|---:|---:|
| Punk Up searched | 2.62 | 2.65 |
| Poffin searched | 1.60 | 1.64 |

Punk Upの勝敗差、rating gradientにも強い追加leverはなかった。このためv28は
ここを推測ルールで再変更せず、測定済みbudgetを維持する。

### 5. Ogerponは専用shellではなくrace + telemetry

271試合の再構成では、Ogerponへ返したturnの93.9%で次のTeal Danceを含めると
こちらをOHKO可能だった。自分ActiveのEnergyを2以下にしても改善は0.4pt。
Grass weaknessによるdeck-level lossであり、policy-onlyの強制ルールを作る根拠が
ない。v28はTeal Ogerponを専用route名で記録するが、動作は現在パイロット由来の
v25 race policyとする。支持される目標はturn 8以前に終えることだけである。

デッキも変更しない。historical censusではexact 60のbestが1220.2に対し、既知の
近傍はHandheld Fan型1066.2、Xerosic型1070.6、Yveltal等5枚変更型1083.5だった。
弱点対策カードを入れて全般のconsistencyを落とす証拠のほうが強い。

### 6. v27から削除したもの

- belief H2/H3 search: 実ラダーboardでoverride 0の死荷重。
- mirror Froslass veto: v24で-72.7 Elo、非因果と判定済み。
- value model: 使われないsearch専用なので同時に削除。

## 検証結果

### Tests and static validation

- 208 unit/regression tests: PASS
- `scripts/validate_agent.py`: PASS
- deck: 60 cards、19 unique、warning 0
- race model / wall model / planner / router / 3 guards: load error 0

### Footprint on all v27 games

Command:

```powershell
.\.venv\Scripts\python.exe scripts\probe_grimmsnarl_v28_footprint.py
```

| metric | result |
|---|---:|
| episodes | 35 |
| evaluated single-pick decisions | 2,755 |
| both-ranker comparable | 2,595 |
| v25-v22 ranker disagreements | 470 (18.11%) |
| v28 final != v22 ranker | 411 (14.92%) |
| ordinary changes | 276 |
| mirror changes | 133 |
| public-wall changes | 2 |
| pre-registered footprint gate | PASS |
| local teacher-forced time | 5.87 sec/game |

保存着手はv27、すなわちほぼv22が生成したため、v22の再現率が高いことはcontrolで
あってv22のaction valueの証明ではない。footprint probeは未選択行動へ勝敗labelを
付けず、各提案後に保存着手で全stateを進めている。

Report: `experiments/grimmsnarl_ml_v28/footprint_v27_run.json`

### Submission archive

- entries: 23
- bytes: 11,998,330
- SHA-256: `db0d4d21381c9aafb82d0b07e5fa7decfcc0b19c493706e4aa32ffaa726ad5f4`
- extracted import smoke: PASS
- archive内で両rankerと全optional componentがload、deck 60枚を確認

## ラダーでの判定方法

測定ラウンドでは v28 と v22 を同時刻に1枠ずつ出す。別日runとの比較、v28単独の
最終rating、素の勝率だけではpromotionしない。

事前登録cell:

1. ordinary race: v22非劣性、点推定は改善
2. wall/tank: v22水準0.54付近を維持
3. second seat: own turn 2のGrimmsnarl ex / Shadow Bullet access
4. confirmed-strength 950+ opponents
5. Ogerpon: route exposureとturn 8以前のrace completion（小標本はratingへ外挿しない）

34試合runの90%幅は約215 Eloである。100〜130未満の差を単発runで断定しない。
得点だけが目的のラウンドは、検証後の同一championを2枠へ入れてmax-of-twoの
期待値+35.5を取る。測定ラウンドと得点ラウンドは分ける。
