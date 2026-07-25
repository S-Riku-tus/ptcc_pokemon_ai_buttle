# CHANGELOG v21

## 方策

- ノココッチの任意ドローを、選択済みKO成立後は原則停止。
- 唯一の攻撃役しかなく次の進化ラインもない場合に限る継戦例外を追加。
- ケーシィ/ノコッチの攻撃後交代と、気絶後の昇格を別スコア関数へ分離。
- 相手の見えている攻撃役から、交代先が次ターン倒される危険を推定。
- 気絶後は攻撃可能な昇格を最優先し、なければ
  ノコッチ > ノココッチ > 安全なキチキギスex > 進化ラインへ変更。
- 実効ダメージまたはダメカンを与えられる攻撃がある場合のENDを禁止。
- ミストエネルギー等で効果0の攻撃は上記禁止から除外。
- 相手残りサイド3枚以下のキチキギスex展開を原則禁止。
- 直前気絶後、3枚ドローが即KOまたは攻撃復帰を作る場合の例外を追加。
- `evolvesFrom`から進化系列を推定し、相手の主力系列の進化前へ将来価値を付与。
- 対象価値へ系列、サイド、エネルギー、どうぐ、HP、エンジン特性を統合。

## 診断

- v21固有カウンタを追加:
  `dudun_continuity_draws`、`progress_attack_end_blocks`、
  `abra_switch_sacrifice_choices`、`ko_promotion_attacker_choices`、
  `ko_promotion_shield_choices`、`fez_late_bench_blocks`、
  `fez_recovery_exceptions`、`core_line_target_bonuses`。
- 実戦リプレイをteacher-forcedで比較する
  `scripts/analyze_alakazam_policy_replay.py`を追加。
- 9つのv21境界テストを`test_v21_runtime_logic.py`へ追加。

## 変更しなかったもの

- デッキ60枚
- LightGBMモデルと特徴量
- 閾値`0.37`
- live ML範囲
- v20までのBoss/KOルート、Hammer、Mist、終局ゲート

## 診断用環境変数

- `ALAKAZAM_V21_TARGET_ROUTES`
- `ALAKAZAM_V21_HAND_GATE`
- `ALAKAZAM_V21_HAND_SURPLUS`（既定2、ノココッチ以外の任意ドロー用）

旧v20の環境変数は互換フォールバックとして残しています。
