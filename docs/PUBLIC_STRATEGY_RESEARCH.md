# PTCG AI Battle Challenge 公開戦略調査と改善方針

## 1. 添付main.pyの診断

添付コードは正常終了しても、戦略的には非常に不利です。

### 致命的な点

1. `option.type`の解釈が誤っています。公式の対応は、`7=PLAY`、`8=ATTACH`、`9=EVOLVE`、`13=ATTACK`、`14=END`です。添付コードは`7=ワザ`としているため、攻撃ではなくカード使用を最優先しています。
2. すべての選択場面を同じ`score(opt)`で処理しています。初期配置、捨て札、サーチ、交代、攻撃対象では、同じカードでも価値が逆になります。
3. 常に`ranked[:maxCount]`を返します。任意で0～複数枚選べる場面でも最大枚数を選び、不要な捨て札や対象選択を行う可能性があります。
4. デッキがカード5種×4枚＋草エネルギー40枚で、コメントにも「例」と書かれています。デッキと判断ロジックの連携がありません。
5. ランダム値を毎回加えるため、失敗リプレイの再現と改善比較が難しくなります。

## 2. 公開されている強い構成

非公開のリーダーボード上位コードは取得できません。そのため、公開Notebook、公式サンプル、公開リポジトリで再現可能な構成を比較しました。

### Mega Lucario ex / Hariyama / Solrock

公開公式サンプルと高レート報告付きの派生版で繰り返し使用されている構成です。

- Mega Lucario exを主力にし、少ないエネルギーから高打点を出す
- Rioluからの進化を優先
- Solrock＋Lunatoneを小型アタッカー／補助として使う
- Hariyamaを非exアタッカーとして残し、Crustleのex無効能力に対応
- SwitchとBoss's Ordersを、事前に作った攻撃計画に合わせて使用

公開ベンチでは、公式ルールベースがランダムエージェントに40勝0敗、別の単純学習方策にも28勝12敗と報告されています。

### Crustle Wall

Crustleの「相手のPokémon exからのダメージを受けない」性質を中心にしたメタデッキです。

- Dwebble→Crustleを4-4で厚く採用
- Koraidon ex、Cornerstone Mask Ogerpon ex、Munkidoriなどを組み合わせる
- 相手がex主体の場合に非常に強い
- ただし、非exアタッカーやBoss's Ordersによるベンチ狙いへの対策が必要

公開対戦ログ分析では、リーダーボード上の調整済みCrustle系が単純な公開テンプレートより強いことが示されています。

### Dragapult ex Tempo Control

Dreepy→Drakloak→Dragapult exの進化ラインと、妨害・展開札を組み合わせた構成です。

- 進化を最優先
- Buddy-Buddy Poffin、Ultra Ball、Rare Candyで盤面を作る
- Crushing Hammerなどで相手のテンポを落とす
- MAIN、捨て札、手札サーチ、ダメージ対象を別スコア関数に分離
- Optional multi-selectでは正の候補だけを選ぶ

## 3. 公開強エージェントの共通点

1. デッキとAIロジックが一体化している
2. `SelectContext`ごとに判断を分けている
3. そのターンの攻撃者・標的・必要エネルギーを先に計画する
4. 特定メタへのカード固有対策を持つ
5. 例外時にクラッシュせず、合法手へフォールバックする
6. 先後を入れ替え、80～100試合以上で比較する
7. 高度なMCTSを入れる前に、ルールベースとデッキの整合性を上げる

## 4. 今回作成した版

今回の`main.py`は、Mega Lucario ex系を採用しました。

理由は次のとおりです。

- 公式サンプルとして公開され、動作実績がある
- カードIDと攻撃計画が公開されている
- 現在のコードからの改善幅が大きい
- Crustle対策としてHariyamaを明示的に使える
- ルールベースのため、リプレイを見て修正しやすい

## 5. 評価方法

最初の提出前に、最低限以下を実施してください。

- 現行チームAIとの100試合（先後50ずつ）
- 公開Mega Lucario、Dragapult、Crustle系との各80～100試合
- エラー率0%
- 攻撃可能なのにENDした回数
- 不要な最大枚数選択を行った回数
- CrustleにMega Lucario exで0ダメージ攻撃した回数

評価結果をCSVで残し、勝率だけでなく失敗タイプ別に修正するのが重要です。

---

## 6. 2026-07-10 メタ更新（重要：上記1〜5の前提は崩壊済み）

公開リポジトリ [wmh/ptcg-abc](https://github.com/wmh/ptcg-abc)（実ラダーepisodeの
日次分析＋top-100パイロットのdivergence miningを継続している参加者）の調査より。

### 現メタ（Elo≥1000, 2026-07-05 episode、使用率/勝率）

1. Grimmsnarl ex: 38.6% / 50.6%（新覇者）
2. Alakazam 非ex 741ライン: 17.5% / 52.3%
3. Kangaskhan ex: 11.6% / 56.9%（Grimmsnarlに82%有利）
4. Cynthia's Garchomp ex: 10.4% / 60.1%（全体最高勝率。二大勢力の両方に有利）
5. 新#1プレイヤー(vibechu, Elo 1195)はSlowkingツールボックス型（要監視）

### 旧前提の失効

- **Mega Lucario ex は絶滅**（6-21で使用率0.4%/勝率46%→消滅）。mega_lucario_v1の
  デッキ選択自体が敗因になっており、パイロット改善では回復不能。
- Crustleも激減。Hariyama対策の価値は消失。
- メタは数日で反転する。提出前に必ず最新episodeデータセット
  （kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD）で分布を再確認すること。

### 得られた知見（wmh/ptcg-abcのlessons learned）

1. デッキ選択がエージェント品質より支配的（ただし単純なデッキ×ルールベースが最良）
2. ローカルsim（cabt含む）はラダー順位を正確に予測しない。回帰検知用と割り切り、
   最終判断は実ラダーA/B（最新2提出が採点対象、5回/日）
3. 40戦のA/Bは±10pt級のノイズ。80戦以上で判断
4. スコアの当てずっぽう調整は禁物。top披露のreplayとのdivergence分析で修正する
5. エネルギー過剰貼りは共有基盤（policy_base.pyのshould_fuel）で構造的に防止

### 本リポジトリの対応（2026-07-10）

- `agents/cynthia_garchomp_v1/`: 現メタ最高勝率アーキタイプ＋mined pilot（主力候補）
- `agents/alakazam741_v1/`: 非ex 1プライズ型（ヘッジ。直接対決ではGarchompに67%勝）
- `agents/_base/policy_base.py` + `generic_policy.py`: 共有基盤（wmh/ptcg-abc由来）
- `vendor/cg/`: ローカル対戦環境（公式wheelのエンジン＋自作互換API。Git管理外）
- `scripts/local_arena.py`: 依存なしのローカルベンチ（Windows/Linux両対応）
