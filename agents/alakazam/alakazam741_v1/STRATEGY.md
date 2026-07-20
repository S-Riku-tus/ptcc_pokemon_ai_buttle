# alakazam741_v1（ヘッジ用第2デッキ）

## コンセプト

全員1プライズの非exデッキ。Dudunsparceの特性でドローを回し、手札を15〜20枚に膨らませ、
Alakazam(741)の **Powerful Hand（手札×20ダメージ）** で300〜400点を置く。
相手のex/Mega ex（2〜3プライズ）を1プライズのポケモンで倒し続け、プライズレースで勝つ。

2026-07-05メタで使用率17.5%の第2勢力（勝率52.3%）。ラダー#1〜2のMajkel1337が
このアーキタイプを使用しており、本実装は同氏の牌表と10,390決定をマイニングした
wmh/ptcg-abc alakazam v3の移植（v3は旧851.5-Elo版との直接A/Bで62%勝）。

## ローカルベンチ（scripts/local_arena.py, 各20〜30戦, エラー0）

- vs cynthia_garchomp_v1: **67%**（直接対決で勝ち越し）
- vs Grimmsnarl(GenericPolicy): 80%
- vs Kangaskhan(GenericPolicy): 100%
- vs Mega Starmie(GenericPolicy): 65%
- vs Alakazam-741(GenericPolicy): 95%
- vs mega_lucario_v1: 90%

## 提出戦略

Kaggleは「最新2提出」が採点対象。**cynthia_garchomp_v1とalakazam741_v1の両方を提出**し、
実ラダーのEloで優劣を判断するのが最適（ローカルsimはラダー順位を正確に予測しない）。

## 出典

[wmh/ptcg-abc](https://github.com/wmh/ptcg-abc) agents/alakazam（v3, 2026-07-07）を移植。
コンペのコード共有ルールを提出前に確認すること。
