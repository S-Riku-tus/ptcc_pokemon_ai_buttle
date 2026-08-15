# Grimmsnarl ML v29 — 対面別エリート方策

作成日: 2026-08-15  
Agent: `agents/grimmsnarl/grimmsnarl_ml_v29`  
親: `grimmsnarl_ml_v28`

## 結論

v28は失敗作ではない。保存35戦では24-11、相手平均860.5、勝率からの実力換算996.0で、v22プールの1008.8と統計的な差がなかった。v26/v27と違い、2898単一選択のうち487手（16.8%）がv22ランカーと異なり、実装した方策は実際に盤面へ届いていた。

ただしv28の実体は、全判断の96.6%をv25のAlphaTCGランカーが担当し、壁だけv22へ切り替える方策だった。壁スイッチは98/2898判断にしか届かない。上位帯で増えるメガミミロップ/ユキメノコとHydrapple exはv25側に残っており、ここがv28に残った明確なルーティング不整合である。

v29は次の限定変更だけを行う。

| 公開対面 | v29の所有方策 | 理由 |
|---|---|---|
| 通常レース、ミラー、フーディン | v25 / AlphaTCG | v28で強かった領域を保存 |
| 既存の壁・Neutralization Zone | v22 / 1220教師 | v28で修正済み |
| メガミミロップ / メガユキメノコ | **v22 / 1220教師** | v22 10-9、v25 1-3 |
| Teal Mask Ogerpon + Applin/Hydrapple | **v22 / 1220教師** | v22 2-1、v25 2-3 |
| 純Teal Mask Ogerpon | v25 / AlphaTCG | 両方負けており、根拠ある改善がない |

新しいモデル、探索、ハードコードした個別手は追加していない。両ランカーはv28と同じく最終実手で履歴を同期し、対面が後から判明しても同じ試合履歴から切り替わる。

## 1. v28が想定レートへ届かなかった理由

### 1.1 レートはまだ収束していない

v28は35 rated games、950以上は7戦だけで、勝率のWilson 95%区間は0.520–0.815である。最終レート968.7だけから「方策が弱い」とは判定できない。対戦相手平均を補正した996.0の方が現時点の実力推定として妥当である。

### 1.2 v28はほぼv25だった

v28でv22壁ランカーが担当したのは98/2898判断だけで、残り2800判断はv25だった。つまり「v25の通常対面の良さ」と「v22の壁性能」を組み合わせる設計は実装どおり動いたが、難対面の分類が壁だけでは狭すぎた。

### 1.3 上位帯で対面構成が変わる

Ogerpon、メガミミロップ/ユキメノコ、Hydrappleの合計は、相手900未満では7.0%だが950以上では31.6%になる。低帯で通常レースに勝って上がった後、上位帯で苦手対面の比率が上がるため、全体勝率の良さがそのまま最終レートへ変換されない。

### 1.4 純Ogerponは方策差ではなく交換レート

Teal Mask Ogerpon exは草、Marnie's Grimmsnarl exは草弱点である。320 HP・2サイドの主戦力が一撃で落ちる。保存プールではv22が1-6（相手平均990.7）、v25が1-2（878.2）で、上位教師へ切り替える根拠もない。

v28の純Ogerpon 4戦をv29で再生した結果、変更は0手だった。これは未修正ではなく、未検証ルールで通常性能を壊さないための意図的な非介入である。

## 2. 採用した介入の根拠

保存480戦からv22とv25の対象セルだけを比較した。

| 対面 | v22 | 相手平均 | 実力換算 | v25 | 相手平均 | 実力換算 |
|---|---:|---:|---:|---:|---:|---:|
| メガミミロップ / ユキメノコ | 10-9 | 982.3 | 1000.6 | 1-3 | 894.3 | 703.5 |
| Hydrapple ex | 2-1 | 1021.4 | 1141.8 | 2-3 | 984.5 | 914.1 |
| 合計 | **12-10** | **987.6** | **1019.3** | **3-6** | **944.4** | **824.0** |

相手レートと先後を固定したロジスティック回帰では、対象2セルのv22ダミーは+186.4 Elo、p=0.244だった。方向と効果量は採用を支持するが、31戦しかなく統計的有意ではない。したがってv29は「最強を証明した版」ではなく、既知の強いセルを狭く移植した検証可能なchallengerである。

ルーティングは公開情報だけを使う。

- Mega Lopunny ex、Buneary、Mega Froslass exの公開でLopunny routeへ入る。
- Hydrapple exの公開、またはTeal Mask OgerponとApplin/Dipplinの同時公開でHydrapple routeへ入る。
- Applin単独ではHydrapple判定しない。無関係なFestival Lead/Dipplinへの誤発火を防ぐ。
- Teal Mask Ogerponが先に見え、後からApplinが見えた場合は、stickyなOgerpon routeからHydrapple routeへ一度だけ昇格する。
- 壁情報は従来どおり全ルートより優先する。

## 3. 採用しなかった案

### 3.1 Adrena-Brain強制使用

盤面状態からの推定では「条件が揃ったのに序盤で使わなかった」30戦が候補だった。しかし実際のMAIN選択肢を数えると、v28はMunkidori能力を提示された108自ターンすべてで使用し、damageありのlive uptakeも108/108 = 100%だった。

したがってこれは改善余地ではない。30戦は「盤面上の必要条件」と「エンジンがその時点で能力を合法提示したこと」を混同した偽陽性だった。v29にはAdrena-Brain overrideを入れていない。

### 3.2 非exをOgerponの前へ強制昇格

一見すると2サイドを守れるが、このデッキの非exは有効なサイドレースを作れず、逃げエネルギーも必要になる。上位同一60のパイロットもOgerponに7-24、他の1100級も0.15–0.31程度で、方策だけで五分にした実績がない。反実仮想結果がない状態で強制ルールにはしなかった。

### 3.3 デッキ変更

既知の近傍デッキの最高値はHandheld Fan型1066.2、Xerosic型1070.6、Yveltal等の5枚変更型1083.5だったのに対し、exact 60の既知最高は1220.2だった。今回のログは「現在の60を捨てる」根拠ではないため、デッキを保存した。

### 3.4 探索の復活

v27のbelief searchは301回検討、23回探索、336 branch評価でoverride 0だった。相手ターンを跨がない探索や未校正の局面価値を再導入しても、今回の対面別証拠より弱い。v29では復活させていない。

## 4. teacher-forcing footprint

v28が到達した36保存episodeを、保存実手で両ランカーの履歴を進めながらv29で再評価した。

| 指標 | 結果 |
|---|---:|
| 単一選択判断 | 2898 |
| v25所有 | 2457 |
| v22所有 | 441 |
| v28実手を再現 | 2832 / 2898 = 97.72% |
| v28から変わる判断 | 66 / 2898 = 2.28% |
| component load error | 0 |
| ローカル再生時間 | 5.50秒/試合 |

対象・negative control別:

| 対面 | 試合 | 評価判断 | v22所有 | v28から変更 |
|---|---:|---:|---:|---:|
| メガミミロップ / ユキメノコ | 3 | 294 | 290 | **62** |
| Hydrapple ex | 1 | 53 | 53 | **4** |
| Ogerpon | 4 | 273 | 98（既存壁型のみ） | **0** |

66差分には同一ターン内の順序差も含まれる。teacher-forcingは「変更が届くこと」と「変更範囲」を証明するが、その手を選んだ反実仮想の勝敗は証明しない。

再現:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_grimmsnarl_v29_policy.py
.\.venv\Scripts\python.exe scripts\probe_grimmsnarl_v29_matchup_routes.py
.\.venv\Scripts\python.exe scripts\probe_grimmsnarl_v28_footprint.py `
  --run data\runs\grimmsnarl\20260815_grimmsnarl_ml_v28_sub55526859 `
  --submission 55526859 `
  --agent agents\grimmsnarl\grimmsnarl_ml_v29 `
  --output experiments\grimmsnarl_ml_v29\footprint_on_v28_run.json
.\.venv\Scripts\python.exe scripts\analyze_grimmsnarl_v16_ability_uptake.py `
  --run data\runs\grimmsnarl\20260815_grimmsnarl_ml_v28_sub55526859 `
  --submission 55526859 `
  --report experiments\grimmsnarl_ml_v29\adrena_option_uptake_v28.json
```

## 最終検証

- 214 unit/regression cases: PASS
- `scripts/validate_agent.py`: PASS
- deck: 60 cards、19 unique、warning 0
- targeted Python files: ruffはローカル環境に未導入のため未実行
- submission archive: 23 entries、11,999,031 bytes
- SHA-256: `3c49cc508ffe71c5ac9baea10ffaf75a1148f42e0b0d8b63b0658640ed7a57ee`
- 抽出後import smoke: agent callable、deck 60、全component load error 0

## 5. ladderでの判定方法

2枠はv28 controlとv29 challengerにする。v27は明確に劣るため残さない。v29を2本同時投入するとペアリング運と方策差を分離できないので、最初の比較では行わない。

一次判定:

1. メガミミロップ/ユキメノコ + Hydrappleの相手レート補正成績。
2. 950以上帯の相手レート補正成績。
3. 通常レース、ミラー、純Ogerpon、壁でv28に対する非劣性。
4. runtime error、盤面全滅、山札切れが増えていないこと。

35戦単独の最終レートだけで昇格させない。最低でも対象対面が複数入るまで走らせ、同時刻のv28と比較する。v29が対象セルを改善せず、通常セルを落とした場合はv28へ戻す。

## 6. 正直な位置づけ

v29は、現時点で作れる中ではv28より論理的に強い候補である。理由は「改善しそうなルール」を増やしたからではなく、実際に強かった上位教師を、実際に差が出た対面だけへ割り当てたからである。

一方、純Ogerponというデッキ構造上の上限は残る。v29を提出前から「必ず最強」「必ず目標レートへ届く」とは主張できない。正しい主張は、通常性能を97.72%保存しつつ、v28の3敗が集中した対象セルへ66判断分の実変更を届けた、測定可能な次のchallengerである。
