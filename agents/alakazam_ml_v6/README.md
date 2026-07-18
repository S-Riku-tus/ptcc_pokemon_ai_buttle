# alakazam_ml_v6

ML v5の2提出・92試合を解析して作成したv6です。

## 主な変更

- 唯一の場のポケモンであるのこっちの特性を絶対禁止
- ゲノセクト1、ラッキーメット1、基本超エネルギー1を削除
- ボスの指令3枚を採用
- ボスは同ターンKOかつActiveより価値が高い対象に限定
- Team Rocket’s Articunoなどの効果保護役を優先
- `ranker_model.json`は変更せず、BossとAbilityはルール専用

詳細は`V5_LOG_ANALYSIS_AND_V6_CHANGES.md`と`VALIDATION_REPORT_V6.md`を参照してください。

## runtime

`main.py`は`fallback_v3.py`を権威ロジックとして使用し、MLはデフォルトでshadow-onlyです。

## 提出用ファイル

- main.py
- fallback_v3.py
- fallback_v12.py
- policy_base.py
- common_runtime.py
- ml_features.py
- ml_runtime.py
- ranker_model.json
- deck.csv
- metadata.json
