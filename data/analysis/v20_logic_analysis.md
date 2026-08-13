# v20 改修に向けた v19 ロジック/ML 分析（Claude の一次分析）

対象: `agents/alakazam/alakazam_ml_v19/`
前提データ: `data/analysis/prize_taking_v19_report.md`（サイド6枚の取り方の多角分析）、
自分ログ52戦（sub 54938555, 30-22）、top40 フーディンログ1,264戦。

このドキュメントは「別の生成AIにもう一度分析させ、最終AIが両者を統合して実装する」ための
一次分析。ユーザーの3要件に対する **現状マッピング / gap / 実装方針 / リスク** を、
コード位置つきで整理する。

---

## 0. アーキテクチャ要点（実装者が最初に知るべき事実）

- エントリ: `main.py` → `HybridRanker.choose(obs, fallback)`（ML, threshold 0.37）。
  `select` が無い局面は純 fallback。
- `HybridRanker` が **上書きするのは `ML_ALLOWED_ACTIONS = {"bench","evolve"}` のみ**
  （`ml_runtime.py:19-35`）。**Boss / attack / trainer / energy / ability / retreat / xerosic /
  hammer はすべて `RULE_ONLY` = 決定論的 fallback**。
  → **今回の3要件（Boss条件・手札最適化・ターゲット優先順位→KO）は 100% fallback 側の改修で完結し、
     ML モデル(`ranker_model.json`)の再学習は不要。** これは実装コストを大きく下げる重要事実。
- fallback は `PolicyEngine._score(o)`（`fallback_policy.py:2222`）で全オプションをスコアリングし
  最大値を選ぶ。プライズ値は `prize_count`（メガ=3 / ex=2 / 単=1）。
- 既にある「KOルート探索」`_active_route_plan()`（`:1604`）は **相手Activeを Alakazam Powerful Hand
  (20×手札) で倒す最小確定手順** を bitmask 探索で求める。だが **対象はActive限定**で、
  ベンチを含む優先順位付けは無い。

---

## 1. 要件① Boss's Orders の絶対条件

### ユーザー要件
> サイド枚数が 2 or 3 枚の時、相手ベンチのHPと自分の与ダメージが足りていて、
> サイドを全て取れる時に Boss を使う（絶対条件）。

### 現状マッピング
- Boss の PLAY スコア: `_score_play_trainer`（`:2701-2711`）
  → 有資格ターゲットが無ければ `-1`、あれば `18500 + min(12000, best//8)`。
  `best = _boss_target_score(p)`。
- `_boss_target_score`（`:1862`）に「winning」概念あり:
  `target_prizes >= len(me.prize)` で `score += 90000`（`:1938-1939`）。
- ただし **PLAY スコア側で `min(12000, best//8)` に圧縮**されるため、勝ち確 Boss でも
  最終 Boss スコアは実質 **~30500 が上限**。lethal Active 攻撃(90000)より低い。
- lethal-Active 分岐（`:2253-2269`）: Activeを今KOできる時、有資格 Boss は `35000+` を返し
  即攻撃を上書きできる（主にロケット団アルセウス系の保護エンジン用）。
- **`_winning_gust_ready()`（`:2606`）が定義済みだが未使用（デッドコード）**。中身はまさに
  「ベンチに今の攻撃でKOでき、そのプライズで勝ち切れる対象がいて、現Active KOでは勝てない」
  ＝ユーザー要件そのもの。

### gap
1. 勝ち確 Boss が **絶対（最優先固定スコア）になっていない**。`best//8` 圧縮で他の高スコア手
   （lethal 攻撃・展開）と競合し、局面次第で選ばれない可能性がある。
2. ユーザーの「2 or 3枚残 × 全取り」という明示ゲートが存在しない。現状の winning 判定は
   `target_prizes >= remaining` のみで、残数を 2/3 に限定していない（実害は小さいが、
   1枚残では単KOで勝てるので Boss 不要／4枚以上残では単発 Boss で全取り不可、の切り分けが暗黙）。
3. `_winning_gust_ready()` が結線されていない。

### 実装方針（案）
- `_score` の MAIN 冒頭に **勝ち確 Boss 絶対ゲート** を追加:
  条件 = `len(me.prize) in (2,3)` かつ あるベンチ `p` について
  `prize_count(p) >= len(me.prize)` かつ `_boss_damage_after_spend(p) >= p.hp`
  かつ `not _effect_prevented(p)`（＝Bossで引きずり出して今のターンにKOし全取り）。
  成立時、その Boss PLAY を **terminal win 級（例: 90000）** に固定し、圧縮 `//8` を通さない。
- `_winning_gust_ready()` を上記ゲートの実体として再利用（残数2/3の限定を足す）。
- 「Active KO でも勝てる」場合は Boss 不要 → 既存の `winning gust ready` は
  `prize_count(opp_active) >= remaining` で false を返すので二重KOの無駄撃ちを防げる（流用可）。

---

## 2. 要件② 手札枚数の最適化（倒したい相手のHP基準・過剰ドロー抑制）

### ユーザー要件
> 手札=命。クセロシキのたくらみ等で手札を削られると Powerful Hand の火力が壊れる。
> だから「倒したいポケモンの残HPを踏まえて手札を最適化」し、むやみに増減させない。

### 現状マッピング
- `_achievable_hand`（`:2455`）= 現手札 + Run Away Draw(+3) + サポ(+1)。
- `_ko_active_reachable`（`:2616`）= `20*_achievable_hand >= opp_active.hp`。
- `_active_route_plan`（`:1604`）= Active を倒す **最小** 手順（action数→デッキ消費→サポ消費で最小化）。
- 致死維持ゲート（`:2253-2276`）: Alakazam が今 lethal の時、手札を減らすプレイ
  （`_hand_delta<0` かつ `20*(hand-1) < opp.hp`）はスコア10に落として攻撃を選ばせる。
- Dudunsparce Run Away Draw（`:2343-2388`）: **「積極ドロー＝アイデンティティ」で基本 15000**。
  停止は (a) `hand>=12 and deck<=14`、(b) `_deck_preserve()`、(c) 勝ち手札×低デッキ のみ。
  コード内コメントが自認: **上位は Run Away Draw 発動が約1/4（top 163 回 vs 我々 622 回）**。

### gap
- **手札の目標が常に「相手Active」しか見ておらず、かつ最小化ではなく最大化に寄っている。**
  `_active_route_plan` は最小手順を出すが、Dudun ABILITY が無条件 15000 で上書きし、
  致死到達後もドローを続ける構造。→ ユーザー指摘・上位挙動・prize分析(過剰KO＝遅い)の三者と矛盾。
- **クセロシキ(自分が撃たれる側)への防御思想が無い**。手札を唯一の火力源にして過剰に積むほど、
  相手 Xerosic 一発で Powerful Hand が崩壊するダウンサイドが増える。現状これを避ける抑制が無い。

### 実装方針（案）
- 「今ターンの優先ターゲット」（要件③で定義）の HP に対する **必要手札** を計算:
  `need_hand = ceil(target.hp / 20)`（Powerful Hand 前提。245/ユンゲラー致死は手札非依存で別扱い）。
- 優先ターゲットが **既に lethal（現手札で到達）** なら、追加のドロー系
  （Dudun / 追加サポ / Poffin）を **抑制**（致死維持ゲートの逆方向: 増やす手も止める）。
  例外: (a) 次ターンのバックアップ育成に必要、(b) デッキ管理上必要、(c) 相手 Xerosic 実出現時の
  **アンチXerosicバッファ**（need_hand + α まで許容）。
- Dudun の無条件 15000 を「need_hand 到達後は低スコア」に条件化。
  ただし後述リスク（cabt のデッキアウトガード後退）を必ず A/B。

---

## 3. 要件③ ターゲット優先順位付け → 最優先から順にKOルート探索

### ユーザー要件
> ターン開始（1ドロー後）時点で「どのポケモンを倒すべきか」の優先度が決まる。
> 優先度 = 自分と相手のサイド枚数比較 + 双方の場/ベンチの構成 から計算。
> 最優先の相手ポケモンを、手札のカード使用や場のやりくりでどう倒すか判断し、
> 無理なら次点の相手ポケモンへ…という順で考える。各カードの使用条件がそこに乗る。

### 現状マッピング
- ターゲット価値: `_target_value`（`:1781`）= プライズ×2200 + エネ×380 + ツール×220 +
  role_bonus + 進化段階 + HP/2。`_boss_role_bonus`（`:1753`）で保護・攻撃役・ex等を加点。
- だが **統合された優先順位リスト（Active∪Bench を1本の順序に並べ、上から順にKOルートを試す）が無い**。
  Active は `_active_route_plan`、Bench は Boss スコアで **別々**に処理。相手のサイド枚数（`opponent.prize`）
  を優先度に使う箇所も限定的。

### gap（最大の構造的欠落）
- ユーザーが望む「優先度順に1体ずつ KO 可否を判定して落ちなければ次点」という
  **明示的なターゲット選択ループ**がエンジンに無い。現状は各行動の局所スコアの総和で暗黙に決まる。

### 実装方針（案）
1. `_kill_priority_list()` を新設: `opponent.active + opponent.bench` の各ボディに
   **優先度スコア** = f(自/相手の残サイド, `prize_count`, `_target_value`,
   相手の対自脅威度, 引きずり出しの粘着性(retreatCost>エネ), 到達可能性) を付与し降順ソート。
   - prize分析の含意を反映: 相手アーキ別に高プライズ体(メガ3→ex2)を優先、対単は長期戦前提、
     vs ex は 2+2+1+1 / 2+2+2 を基準線に。
2. `_active_route_plan` を一般化した `_ko_route(target)` を新設:
   任意ターゲットへの最小確定KO手順を返す（Active=直接、Bench=Boss+攻撃で Powerful Hand -20 と
   サポ消費を織り込む）。
3. 優先度順に `_ko_route` を試し、**最初に到達可能なターゲット**を「今ターンのプラン」に確定 →
   そのターゲットHPで要件②の `need_hand` を決める。到達不能なら次点へ。
4. 各カードのスコアは「確定したプランを実現する行動か」で加点（既存の
   `_active_route_action_score` の一般化）。

---

## 4. 横断的な注意（実装者向け）

## 5. 具体的な実装優先度（Claude の推奨順）

1. **要件①（Boss 絶対ゲート＋`_winning_gust_ready` 結線）** — 最小変更・最も安全・勝ち筋直結。
2. **要件③（優先順位リスト＋汎用KOルート）** — 中核。②の前提。
3. **要件②（need_hand ベースの過剰ドロー抑制）** — 効果大だが cabt 後退リスクが最も高い。A/B 厳重。
</content>
</invoke>
