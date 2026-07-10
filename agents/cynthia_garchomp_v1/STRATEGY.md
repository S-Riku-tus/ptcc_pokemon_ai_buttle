# cynthia_garchomp_v1

## なぜデッキを乗り換えたのか

2026-07時点のラダーメタでは、Mega Lucario exは絶滅済みです（6-21時点で使用率0.4%・勝率46%、
その後消滅）。現在のトップメタ（Elo≥1000, 7-05 episodeデータ）は次のとおりです。

| アーキタイプ | 使用率 | 勝率 |
| --- | --- | --- |
| Grimmsnarl ex | 38.6% | 50.6% |
| Alakazam(非ex, 741ライン) | 17.5% | 52.3% |
| Kangaskhan ex | 11.6% | 56.9% |
| Cynthia's Garchomp ex | 10.4% | **60.1%（全体最高）** |

Cynthia's Garchomp exは二大勢力の両方（Grimmsnarl 68% / Kangaskhan 60%）に有利で、
現メタで最も立ち位置が良いデッキです。

## デッキの役割

- Gible(379)→Gabite(380)→**Garchomp ex(381, 330HP, 逃げ0)**：主軸
- Gabite特性 **Champion's Call**：毎ターン無料でCynthia'sポケモンをサーチ（進化を急がず維持する）
- **Corkscrew Dive [F]=100＋手札6枚まで補充**：エネ1個で打てる主力ワザ
- **Draconic Buster [FF]=260（全エネ破棄）**：Corkscrewで倒せない相手への必殺用のみ
- Roselia(341)→Roserade(342)：特性で自分のワザ+30（常時パンプ）
- Spiritomb(387) Raging Curse：自ベンチのダメージ×10をばら撒く第2アタッカー
- Rock Fighting Energy(20)：{F}＋ワザ効果を防ぐ。前のGarchompに優先貼り
- Power Weight(1173)：Garchomp 330→400HP

## パイロットの要点（トップ100パイロットの実対局からマイニング済み）

1. **Garchompへの進化は急がない**。GabiteのChampion's Callが毎ターンのサーチ源。
   エネが付いたGabiteだけを、攻撃するターンに進化させる。
2. ベンチは広く展開（全員1プライズ＋ベンチの被ダメがSpiritombの火力になる）。
3. エネルギーは過剰に貼らない（policy_baseの汎用エネ規律で構造的に過貼り不可）。
4. Boss's Ordersは「多プライズをgust-KOできる時だけ」。
5. 先攻を取る。

## ローカルベンチ（scripts/local_arena.py, 各40戦, エラー0）

- vs mega_lucario_v1（旧エージェント）: **72.5%**
- vs Grimmsnarl(GenericPolicy): **100%**
- vs Kangaskhan(GenericPolicy): **95%**
- vs Alakazam-741(GenericPolicy): **85%**
- vs Mega Starmie(GenericPolicy): **62.5%**

注意: ローカルsimはラダー順位を正確には予測しない（相手パイロットが本物のtop-100より弱い）。
最終判断は実ラダーのA/Bで行うこと。

## 出典・注意

実装は公開リポジトリ [wmh/ptcg-abc](https://github.com/wmh/ptcg-abc)（GarchompPolicy、
nasuo445の12,693 MAIN決定をdivergence miningした成果）をベースに移植したもの。
デッキリストはラダー上位nasuo445の公開episodeと同一。コンペのコード共有ルールに
抵触しないか、提出前にルールを確認すること。

## 次の改善候補

1. 毎日episodeデータでメタを再確認（メタは数日で反転する。Slowkingツールボックスが新#1）
2. 実ラダーでのA/B（最新2提出が採点対象、5回/日）
3. 負け対局のreplayからSelectContext別のdivergence分析
