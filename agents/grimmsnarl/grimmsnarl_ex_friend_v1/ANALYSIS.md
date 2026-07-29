# grimmsnarl_ex_friend_v1 静的分析

対戦データなしの**コード読解のみ**による分析（2026-07-29）。
比較対象は自分の [marnies_grimmsnarl_ex_v7](../marnies_grimmsnarl_ex_v7/)。

## 1. 出所

main.py 冒頭のヘッダより:

- 自称バージョンは **v5**（`'version': 'grimmsnarl_v5'` としてログにも出る）。
  このディレクトリ名の `v1` は当リポジトリ側の受領番号であり、友人側の版数とは無関係
- **高ランクAgent (jneums) の対戦ログ3件から60枚と行動順を抽出して再現**したもの。
  「1000ログ行動模倣改善」とあり、模倣学習ではなくログから手作業でルール化した実装に見える
- 随所のコメントが "the winning log used ..." / "The high-rank agent selected first ..." と
  ログ由来の根拠を明示している

つまりこれは**特定の上位エージェントの再現実装**であり、独立に設計された戦略ではない。

## 2. アーキテクチャの差（最重要）

| | friend_v1 | marnies_grimmsnarl_ex_v7 |
| --- | --- | --- |
| 依存 | **標準ライブラリのみ**（json / os / time / dataclasses / typing） | `cg.api` + `policy_base` |
| 観測の扱い | 生 dict (`obs_dict.get('current')`) | 型付き `Observation` |
| 構造 | モジュール関数69個のフラット構成 | `GrimmsnarlPolicy(BasePolicy)` クラス |
| 行数 | 1415 | 1603 |

**この差は移植コストに直結する。** friend 側は `cg.api` を一切使わず生 dict と
数値定数（`CTX_*` / `AREA_*`）を直接叩いているため、v7 のロジックをそのまま
持ち込む／逆に持ち出すことはできず、どちらかの観測レイヤに書き直す必要がある。

## 3. friend にあって v7 にない: 相手アーキタイプ適応

friend の最大の特徴。[main.py:168](main.py#L168) `_opponent_archetype()` が
相手の場・トラッシュ・`_SEEN_OPPONENT_IDS`（試合中保持）から
`'alakazam'` / `'mirror'` / `'unknown'` を判定し、展開方針を切り替える。

| | vs alakazam | mirror | unknown |
| --- | --- | --- | --- |
| オーロンゲライン目標数 | **2** | 3 | 3 |
| マシマシラ目標数 | 2 | **3** | 2 |
| ユキワラシ目標数 | **2** | 0（8ターン目以降1） | 1 |
| Froslass ベンチ評価 | **41000 / 52000** | 26000 / 30000 | 26000 / 30000 |
| Snorunt ベンチ評価 | **36000** | 20000 | 20000 |

加えて Alakazam ライン (741/742/743) への狙い撃ち加点:
ベンチ攻撃プラン +9000 / アクティブ攻撃 +10000 / Boss's Orders 対象 +9500。
[main.py:389](main.py#L389) `_weak_to_dark()` は Alakazam ラインを
**ハードコードで悪弱点扱い**してダメージ計算を底上げする。

戦略の骨子は **Froslass で進化前の Abra/Kadabra をベンチごと落とす**こと。
ミラーでは自分の特性ポケモンも巻き込むため Froslass を遅らせる、という
非対称な扱いも入っている。

> **評価上の注意**: 自分の Alakazam 系（`alakazam_ml_v*`）と当てた勝率は、
> 友人エージェントの汎用的な強さではなく「Alakazam 対策の仕上がり」を測る数字になる。
> 汎用性能を見たいなら `unknown` に落ちる相手（Alakazam でもミラーでもないデッキ）で回すこと。

## 4. v7 にあって friend にない

キーワード出現数の比較（friend / v7）:

| 機能 | friend | v7 |
| --- | --- | --- |
| ウォール検出 (`wall` / `WALL`) | **0** | 40 |
| アタッカーETA (`ETA` / `_eta`) | **0** | 20 |
| `appearThisTurn` 考慮 | **0** | 2 |
| Dodge / Hide 一時無敵 | **0** | 13 |
| fast_race ギア | **0** | 9 |

friend はゼロダメージ攻撃を `-18000` で忌避するだけで
（[main.py:627](main.py#L627)）、「殴れない相手が居座ったときにどうするか」
という v7 が重点的に潰した領域の判断機構を持たない。
Mega Lucario のような高速デッキへのギアチェンジもない。

**両者は補完関係にある。** friend = 相手適応、v7 = 盤面の詰み回避。

## 5. 気になる点

### 5.1 診断フックがない → 計測の盲点

friend の main.py に `_DIAG` / `DIAG` が**存在しない**（v7 にはある）。
`scripts/agent_loader.py` の `get_agent_diag()` はこれを読むため、
self_play が出す `policy_fallback` / `obs_fallback` は friend 側について
**「0 だから健全」ではなく「計測されていないので 0」**である。

さらに [main.py:1378](main.py#L1378) の `agent()` は全例外を握りつぶし、
`list(range(min(required, option_count)))` という**先頭から機械的に選ぶ
フォールバック**を返す。この経路に落ちてもエンジン側は合法手として受理するため、
エラーにも違法手にもカウントされない。`AGENT_LOG=1` を立てない限り不可視。

self_play の `errors` / `illegal` はエンジン由来なので信頼できるが、
**「例外を吐きながら適当な手を返し続けている」状態は検出できない**。
本格的に評価する前に `AGENT_LOG=1 AGENT_LOG_PATH=... ` で回して
`error` レコードの有無を確認すべき。

### 5.2 到達不能な死にコード

デッキに入っていないカードの分岐が生きている:

- [main.py:1256](main.py#L1256) `YVELTAL` (689) — 攻撃スコア分岐
- [main.py:1258](main.py#L1258) `BUDEW` (235) — 攻撃スコア分岐
- [main.py:605](main.py#L605) `YVELTAL` — アタッカー加点 +600

定数のみ定義され未使用: `HANDHELD_FAN` `AIR_BALLOON` `WONDROUS_PATCH`
`TATSUGIRI` `MARNIES_MORPEKO`。

別のデッキリストから移植された痕跡。動作上は無害だが、
**コードがデッキから乖離している**ことの表れなので、改造の起点にするなら先に掃除したい。

### 5.3 整合性チェック（問題なし）

- main.py 内の `DECK` 定数と `deck.csv` は**多重集合として完全一致**（60枚）
- `scripts/validate_agent.py` 通過（60枚 / 19種 / 警告なし）

## 6. その他の設計判断

- **先攻を取る**（[main.py:1196](main.py#L1196)）。「高ランクAgentが選択権を得たとき先攻を選んだ」ため
- マリガンは受けない (`CTX_MULLIGAN` → `want_yes = False`)
- パンクアップは5枚選び、現アタッカー2 / 後続2 / 第3系統1 に分散
- スパイクタウンジム・ポケパッド・ペトレルを**攻撃前**に使って盤面を補充
- マシマシラでダメカン移動 → シャドーバレット180 + ベンチ30 で KO を作る

## 7. 次にやるなら

1. `AGENT_LOG=1` で回し、例外フォールバックが発生していないか確認（5.1）
2. 汎用性能を測るなら Alakazam / ミラー**以外**の相手と当てる（3の注意点）
3. ロジックのみを比較したいなら `deck.csv` を揃えた検証用コピーを作る
   （2枚差の内訳は [metadata.json](metadata.json) 参照）
4. v7 に取り込む価値があるのは**アーキタイプ適応の枠組み**。ただし観測レイヤが
   違うので `cg.api` ベースへの書き直しが必要（2）
