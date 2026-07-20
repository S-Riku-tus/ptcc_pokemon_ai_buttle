# CHANGELOG — alakazam741_v9_top8_core

## 9.0.0 (2026-07-14)

初版。上位8チーム・793試合分析の汎用コアを再現。デッキはv8と完全に同一の60枚（ロジック差のみを測定するため）。

### 追加

- **明示的なターン状態機械**: 毎MAIN選択で `SETUP` / `PRESSURE` / `RECOVER` / `LOCKED` / `ENDGAME` から一つの主要状態を決定（`_classify_state`）。複数のBooleanモードを同時成立させない。状態は役割の充足で判定し、盤面数を固定目標にしない。
- **攻撃予約 (`_attack_reserved` / `_compute_attack_plan`)**: 提示中の最善>0ダメージ攻撃を予約。攻撃前候補は `_preserves_attack`（予約と現在KOを保持するか）と `_improves_plan`（4つの許可カテゴリ）でゲートし、許可された候補のみ攻撃より上のTierへ。ENDは攻撃予約中ブロック。
- **動的山札管理 (`_effective_deck` / `_turns_to_win` / `_turns_to_deckout` / `_optional_spend_ok`)**: 実効山札=山札+ノココッチ復帰(≈2/体)+聖なる灰復帰(最大5)。任意ドローは「現在KOを作る」「最初の後続を確定する」「turns_to_deckout > turns_to_win を保つ」時のみ許可。
- **RECOVER専用の優先順位**: アメ/ユンゲラー再建・夜のタンカ・聖なる灰・エネルギーを優先し、スタジアム/妨害/5体目を後回し。
- **LOCKED専用処理 (`_score_locked`)**: 改造ハンマーで即解除して同ターン攻撃できる時だけ解除、解除不能なら任意ドロー・0ダメージ攻撃を停止。
- **診断の拡張 (`_DIAG`)**: 選択状態・攻撃予約・攻撃・フーディン攻撃・0ダメージ攻撃・攻撃可能END・退避・ノココッチ特性・最後の1体特性ブロック・攻撃前行動を集計。

### 削除（v8から）

- 固定 `LOW_DECK_COUNT = 6` と `_low_deck()`。
- 固定フロア `_deck_floor() = max(8, サイド+3)` と `_deck_spend_ok` の固定閾値。
- `_deck_preserve()` の大手札フロア。
- ノココッチ特性の固定停止 `手札>=12 かつ 山札<=14`。
- 盤面数ベースのなかよしポフィン制御（役割ベースへ置換）。

### 不変（v8から意図的に維持）

- 60枚デッキ（byte一致）、同梱 `policy_base.py`（汎用エネルギー規律・PrizeTracker・フォールバック・make_agent）、Powerful Hand=20×手札 の打点モデル、Hyper Aromaの3枚セット選択、効果防止（盤面全体含む）の一般判定。
- ボスの指令なし・対面別分岐なし。
