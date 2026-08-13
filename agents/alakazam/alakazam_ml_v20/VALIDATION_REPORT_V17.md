# Validation report v17-minimal


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
