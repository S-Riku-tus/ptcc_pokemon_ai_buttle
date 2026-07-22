# Validation report v17-minimal

## 対戦ゲート

| 対戦 | v17-minimal結果 | 目的 |
|---|---:|---|
| vs alakazam_ml_v15 | 520-480 / 1,000 | Champion通常系を下回らないこと |
| vs 修正前v16ベースv17 | 524-476 / 1,000 | 土台変更の直接比較 |
| vs marnies_grimmsnarl_ex_v6 | 249-151 / 400 | 専用制限削除後の安全性 |

2,400戦すべてで引き分け、agent例外、違法選択、policy fallback、observation fallbackは
0でした。対オーロンゲの62.3%はv15対照63.2%と近く、広範な専用ベンチ制限を外した
ことによる大きな退化は見られません。

## 実公開ログ再生

- episode 87405053 / Kinf11
  - Splashing Dodge表を`TO_ACTIVE`から次のMAINまで保持。
  - v16実戦のPowerful Hand option 20ではなく、Boss's Orders option 7を選択。
  - `temporary_immunity_heads=1`、`boss_effect_lock_escapes=1`。
- episode 87413112 / Project Mew
  - Splashing Dodge表を同一観測内で認識。
  - v16実戦のPowerful Hand option 13ではなく、Enriching Energy option 3を選択。
  - `temporary_immunity_heads=1`、`temporary_immunity_blocks=1`。

## 静的・Goldenゲート

- pytest: 72 passed
- `py_compile`: `fallback_policy.py` / `ml_runtime.py` / `main.py`成功
- `validate_agent.py`: 60枚、警告0
- deck SHA-256:
  `57c7d4800cfc0f36581077a40b24912d33056cafcc14cca3783094ce6c122bfe`
- ranker SHA-256:
  `22f41bfa04b4224c566d74d2642f4d8703fa36448dd815cc9b45c61c759e0bbb`
- deck/rankerはv15とbyte単位で同一。
- v15の復元・検索主要9関数はAST単位で同一。
- 攻撃不能の記憶・解除、一時無敵の表/裏/期限/Active離脱、即時KOピボット、完成/未完成
  キチキギス、1体/2体フリーザー、進化したロケット団Boss経路をGoldenテスト化。

## 注意

ローカルエンジンの対戦分布は公開ラダーと異なるため、公開レート上昇は保証しません。
また、フリーザー対策は実公開盤面とGolden状態で検証しており、専用デッキとの大量対戦
結果ではありません。
