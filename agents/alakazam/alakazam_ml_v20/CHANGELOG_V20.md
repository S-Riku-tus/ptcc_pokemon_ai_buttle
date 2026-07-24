# CHANGELOG v20

## ルール変更

- `_terminal_boss_targets()`と`_terminal_boss_gate_score()`を追加。
  残り2/3サイドの合法な勝ち確Bossだけを全MAIN行動より優先する。
- `_target_priority_list()`を追加。ターゲット価値と到達可能性を分離した。
- `_active_route_plan()`の本体を`_ko_route_plan(target)`へ一般化。
  BenchルートにはBoss、手札1枚、Supporter枠を必須コストとして加えた。
- `_chosen_ko_plan()`で優先順に確定KOルートを探索し、無理なら次点へ進む。
- Hammer対象を`planned_hammer_target_key`でルート対象へ束縛した。
- `_attack_hand_required()`と`_guaranteed_hand_delta()`を追加。
  Dawnのnet handを`+2`へ修正した。
- Run Away Draw、Enriching Energy、Dawn/Hilda、進化ドローの評価を
  選択ターゲットに接続。準備完了かつ必要手札+5以上の時だけ抑制する。
- route tie-breakを最大ダメージ優先から最小overkill優先へ変更。

## 診断

- terminal Boss、target route、target到達後draw blockのカウンタを追加。
- `ALAKAZAM_V20_TARGET_ROUTES`と`ALAKAZAM_V20_HAND_GATE`を追加。
  どちらも既定値`1`で、段階A/B用に`0`へ戻せる。
- `ALAKAZAM_V20_HAND_SURPLUS`で抑制閾値を診断可能。既定値は`5`。

## 変更なし

- 60枚デッキ
- `ranker_model.json`
- `ml_features.py` / `ml_runtime.py`
- ML閾値`0.37`と`ML_ALLOWED_ACTIONS={"bench", "evolve"}`
