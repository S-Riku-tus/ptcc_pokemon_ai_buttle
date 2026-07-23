# alakazam_ml_v18

公開版v17（submission `54906373`、39勝21敗、レート833.1）を土台に、
同一ターンの確定行動を最終攻撃まで比較するChallengerです。

デッキ、LightGBMモデル、特徴量、MLの適用範囲はv17から変更していません。
変更対象は決定論的な戦術ロジックだけです。

## 主な変更

- Hammer、進化、リッチエネルギー、ノココッチ、Dawn/Hildaを含む合法手ルート計画。
- Activeの高サイドKOが確定する場合、低サイド対象へのBossを禁止。
- 相手の最後の場のポケモンを倒せる場合、追加展開をせず即攻撃。
- Activeノココッチから勝利攻撃へつながる場合、特性を最優先。
- ノココッチ特性が使えない／使わない理由を診断値として分離。

## 検証

- pytest: 82 passed
- v18 vs v17: 505-495 / 1,000戦
- 最終hardening後 smoke: v18 99-101 v17 / 200戦
- v18 vs v15: 203-197 / 400戦
- v18 vs オーロンゲv6: 247-153 / 400戦
- クラッシュ、不正手、policy/observation fallback: 0

ローカル比較は非劣化の確認であり、公開レート上昇の証明ではありません。
実装根拠は`ANALYSIS_V18.md`、変更一覧は`CHANGELOG_V18.md`、
検証詳細は`VALIDATION_REPORT_V18.md`にあります。

```powershell
uv run pytest agents\alakazam\alakazam_ml_v18 -q
.\.venv\Scripts\python.exe .\scripts\local_arena.py `
  alakazam_ml_v18 alakazam_ml_v17 --games 1000 --seed 1811 --quiet
```
