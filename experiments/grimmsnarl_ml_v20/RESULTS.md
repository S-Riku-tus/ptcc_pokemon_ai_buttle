# Grimmsnarl ML v20

## 結論

v20 は v19 を土台にしつつ、v18 のような「安全処理を足しただけ」の版ではない。822特徴・575木の v19 ranker を、攻撃継続性を表す20特徴を追加した842特徴・513木のモデルへ実際に再学習した。勝利試合4倍の結果依存重みは廃止し、公開観測だけで定義した難しい局面を2倍にする方式へ置き換えた。

ラダー検証はユーザー指定により実施していない。また、過去版とのローカル対戦は提出後の強さを示さないため、モデル選定・昇格判断・本レポートの強さの根拠には一切使用していない。したがって、v20 がラダーで v19 より強いと確定したとは主張しない。提出前に確定できるのは、学習・実行入力の一致、保存盤面での発火、合法性、安全層、パッケージ整合性までである。

## Alakazam 後攻の原因を再調査

提出済み v16-v19 は Alakazam 後攻で 7/18、同じ60枚の field は 114/162 だった。内訳を選択ノードまで掘ると、単純な Punk Up の配分不足という説明は成立しなかった。

| 指標 | v16-v19 | field |
| --- | ---: | ---: |
| ready Active がある Punk Up で将来ラインへ配分 | 100.0% | 97.6% |
| Grimmsnarl ex 進化先が Active | 100.0% | 70.6% |
| ready Grimmsnarl が提示された昇格で選択 | 88.9% | 100.0% |

エネルギー総量ではなく、序盤にどの個体を Active として育て、次の攻撃役を何手で用意できるかが欠落していた。episode 91950626 では、自分で退却した直後に ready Grimmsnarl を提示されながら3エネ Morgremを昇格しており、具体的な1件も特定した。この局面は既存 `attack_access.py` の合法入力掃引で修正されることも確認した。

## 元の5提案への対応

1. **v19 を基準に固定**: デッキ60枚と v19 の安全層を維持した。モデルに実体のなかった teacher route pin は削除し、対面情報で ranker を切り替えない。
2. **難しい局面の再学習**: 勝敗結果による4倍重みを廃止した。攻撃遅延、次アタッカー欠落、Punk配分、ready昇格、壁復帰、mirror終盤という公開観測局面だけを2倍にして LambdaRank を再学習した。
3. **backup Grim ETA**: Active、最良backup、2体目の完成ETA、候補行動前後のETA、ready Active/backup生成、単独アタッカー危険など20特徴を追加した。対面専用hard ruleにはしていない。
4. **wall の preserve + fund + finish**: 既存 `wall_break.py` がすでに進化拒否、手貼り/Punk Upによるbreaker育成、退却・昇格、攻撃完了を実装していたため、重複する広いruleは追加しなかった。全合法入力掃引で実際に拘束することを再確認した。
5. **2ターン Prize Planner**: exact-60 mirror、初回Shadow後、どちらか残り3サイド以下、ranker生スコア差0.08以内だけに限定した tie-breaker を追加した。観測済みダメージ、Froslassのcheckup、次のShadow、最大1回のAdrena-Brain、Boss経路だけを数え、2ターンのサイド上限が厳密に増える場合だけ変更する。即時KO最大化は最後に適用されるため、確定サイドを捨てない。

## モデル学習

コーパスは1,238 episode、95,664 decision、466,463 candidate row。教師ID・対面デッキ・相手手札を特徴へ入れず、教師ごとの時系列 train/validation/test 分割を維持した。

| 候補 | hard-state重み | 木 | validation Top-1 | test Top-1 | test MAIN Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v20 hard1 | 1x | 629 | 80.41% | 80.23% | 73.74% |
| **v20 hard2** | **2x** | **513** | 80.19% | **80.47%** | **74.26%** |
| v20 hard4 | 4x | 841 | 80.17% | 80.39% | 73.69% |
| v19 | 勝利trajectory 4x | 575 | 80.13% | 80.06% | 73.14% |

hard2 は v19 比で test Top-1 +0.41ポイント、判断の中心である MAIN は +1.12ポイント。4倍は局面を過度に強調して MAIN が悪化したため不採用にした。なお、複数候補の比較に test 指標も参照しているので、これを未知ラダーに対する独立した性能保証とは扱わない。

選択モデルの test 局面別 Top-1 は、攻撃遅延80.71%、次アタッカー欠落80.27%、Punk配分92.82%、ready昇格97.04%、壁復帰80.82%、mirror終盤82.78%だった。

## 学習と実行の同値性

新20特徴は v19 コーパス内の公開観測プリミティブから決定的に再構成した。初回監査では、v19 コーパスが effect serial を保持していないことにより `candidate_punk_targets_trigger` が13,776候補中2件だけ実行時定義とずれていた。学習側だけに存在しない精度を持たせないよう、実行時も「Punk Up中、このターン現れたGrimmsnarl」という学習可能な定義へ統一した。

修正後、提出済み v19 の mirror/Alakazam 27戦、2,295判断、13,776候補、20特徴を再照合し、差分は **0** だった。

## 保存盤面での実効性

v19 提出ログの mirror/Alakazam 27戦・2,446決定を保存行動で進める teacher-forced 再生では、モデル更新が172判断を変更し、27戦すべてに届いた。したがって v20 は v18 のような実質同一方策ではない。

2ターン Planner は99 eligible promptのうち2件だけがスコア帯に入り、1件だけ変更した。episode 91940351、turn 11 の Bench-30選択で、110 HP Munkidoriではなく70 HP Impidimpへ30を置き、次のShadow/Adrenaで1サイドへ到達する経路を選んだ。エラーは0。広く上書きするruleではなく、意図した終盤tie-breakerとして実際に拘束している。

安全層は v16-v19 の162戦、14,112 single-pick判断、76,526合法入力を全掃引した。

| 層 | 変更される合法入力 | 拘束prompt | 保存行動への実変更 |
| --- | ---: | ---: | ---: |
| attack access | 3,943 | 3,162 | 3 |
| wall break | 308 | 122 | 2 |
| immediate mirror prize | 148 | 43 | 0 |
| 統合pipeline | 4,389 | 3,314 | 5 |

immediate mirror prize は保存行動を変えなかったが、任意の誤った合法入力を入れる掃引では148入力を修正するため、到達不能コードではない。rankerがすでに正しく選ぶ通常時には不介入となる安全弁である。

## 提出前検証

- v20 全281テスト通過
- `validate_agent.py` 通過: 60枚、19 unique、warning 0
- model: 842特徴、513木、SHA-256 `38435a79d31c999e0dab4283c4e928b9df5b9a30b4b349f47cc78edfb7da3983`
- replay feature error 0、horizon error 0
- 提出アーカイブを展開後に再検証: 60枚、842特徴model読込、全load errorなし、末尾callable=`agent`
- 提出物 SHA-256: `1E7B8A3C88725498EC853BFB17CD3726CCB4C3A8566EEB4A18EB005FBE14117A`
- ラダー未実施（ユーザー指定）
- 過去版との対戦結果は未使用・評価対象外

## 主な成果物

- `agents/grimmsnarl/grimmsnarl_ml_v20/`: 提出agent
- `data/ml/grimmsnarl/processed/corpus_v20_current.npz`: 842特徴コーパス
- `experiments/grimmsnarl_ml_v20/train_v20_hard2_perteam.json`: 採用モデル学習結果
- `experiments/grimmsnarl_ml_v20/route_audit.json`: Alakazam後攻の経路監査
- `experiments/grimmsnarl_ml_v20/feature_parity.json`: 学習/実行特徴同値性
- `experiments/grimmsnarl_ml_v20/footprint_v19_mirror_alakazam_direct.json`: 保存盤面footprint
- `experiments/grimmsnarl_ml_v20/guard_legal_sweep.json`: 全合法入力掃引
- `artifacts/grimmsnarl_ml_v20_submission.tar.gz`: v20提出物（`artifacts/submission.tar.gz` と同一）

最終的な強さの判定は v20 を実際に提出した結果で行う。今回のオフライン結論は「原因候補に届くモデル変更が実際に広く発火し、狭い終盤Plannerも1件拘束し、学習/実行不一致と不正選択がない提出候補になった」までである。
