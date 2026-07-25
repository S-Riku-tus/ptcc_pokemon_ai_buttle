# CHANGELOG v22

## Runtime

### `fallback_policy.py`

- `V22_HAND_TARGET=13`を追加
- `V22_CONTINUITY_DECK_BUFFER=4`を追加
- `V22_FEZ_PROGRESSIVE_BUILD`を追加
- `_continuity_draw_needed()`を追加
- `_optional_hand_growth_needed()`を追加
- ノココッチ停止条件を`backup_eta`基準へ変更
- 継戦用ノココッチ特性を確定KOより上へ設定
- 循環停止後のノコッチ再展開を強化
- 高手札・後続完成時のHilda/Dawn/Poke Pad/Rich任意消費を抑制
- `_fez_progressive_goal()`を追加
- `_fez_progressive_build_allowed()`を追加
- キチキギスexの段階的給エネを追加
- キチキギスexのベンチ100KOを気絶後の攻撃可能昇格として認識
- v22診断カウンタを追加

### 変更なし

- `deck.csv`
- `ranker_model.json`
- `ml_runtime.py`
- `ml_features.py`
- ML threshold・ライブ担当範囲

## Tests

- `test_v22_runtime_logic.py`を追加
- 後続未完成時のKO前循環
- 後続完成時の循環停止
- ノコッチ再展開
- キチキギス段階育成
- 確定KO保護
- 本線未完成時のキチキギス給エネ禁止
- ベンチKO可能キチキギス昇格
- 13枚手札目標
- 13枚超でも後続未完成なら循環継続
- 旧テストの外部兄弟フォルダ依存をSHA/AST固定値へ置換し、完全版単体で実行可能に変更
