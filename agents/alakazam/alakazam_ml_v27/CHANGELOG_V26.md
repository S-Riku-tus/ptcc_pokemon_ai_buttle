# v26 変更一覧

親は `alakazam_ml_v25`（ラダー 57戦 36勝21敗）。デッキ・行動ランカー・標的ランカーは未変更。

## 盤面body下限

- `_body_floor_critical()` / `_body_floor_play_score()` / `_offered_body_add()` を追加。
- 盤面が 1 体以下のとき、body を増やす手を 60000〜62000 に固定し、致死維持ゲートより前で評価する。
  優先順: ポフィン > テレパシックエネルギー（超ポケモンが場にいる時） > ケーシィ/ノコッチ >
  ポケパッド > フェザンディピティ ex > シェイミ。
- 山札に対象が残っていないポフィン／ポケパッド／テレパシックは、この昇格から除外する。
- 盤面 1 体以下で body を増やす手が残っている間は END を選ばない（-2000）。
- ゲーム勝利になる攻撃だけは従来どおり最優先のまま。

## ノココッチ (Run Away Draw)

- 「唯一の body なら使わない」に加えて、**2 体 → 1 体になる循環も拒否**する。
- 例外: 勝利ピボット、同ターンKOルートに組み込まれた循環、低山札ピボット、
  この手番で body を戻せる手（たねポケモン／ポフィン）が提示されている場合。
- 純粋なドローと、攻撃者昇格だけの入れ替えには適用される。

## サポート

- 場にケーシィ・ユンゲラー・ノコッチが 1 体もいないとき、ヒルダを 2000 へ降格し、
  ベンチに空きがあればヒカリを 19000 へ昇格する（進化サーチが空振りする盤面の是正）。

## オーロンゲ (Marnie's Grimmsnarl ex) 対面

- `EVOLVES_FROM_INDEX` を追加し、`_opponent_board_bench_damage()` で相手盤面の
  **進化先まで**ベンチ打点を先読みする。
- `_opp_threatens_bench()` の判定窓を 2 発 → 3 発（ユキメノコ在場なら 4 発）に拡張。
  ただし露出している非ルールボックスのベンチが 2 体以上、かつ自盤面 3 体以上のときだけ真。
- `CHECKUP_COUNTER_ABILITY_IDS` をカードテキストから自動抽出（現環境はユキメノコのみ）。
- `_checkup_counter_engine_upgrade()` を追加。サイドが同数以上で、こちらの特性持ちが
  2 体以上課税されているとき、active_ko 支配則の `role_upgrade` として扱う。

## 診断カウンタ

`v26_body_floor_plays` / `v26_body_floor_end_blocks` / `v26_dudun_body_floor_blocks` /
`v26_hilda_dead_search_demotions` / `v26_dawn_body_searches` / `v26_checkup_engine_gusts`

## 継承テストの更新

v26 が意図的に上書きした 2 件だけ、理由付きで期待値を更新した。

- `test_active_dudunsparce_may_cycle_when_a_ready_body_remains`: 2 体からの昇格目的の循環は
  拒否（body を戻せる場合は従来どおり 14000）。
- `test_shaymin_is_only_benched_against_attack_spread`: 露出 2 体・盤面 3 体以上を満たす形へ更新し、
  満たさない薄い盤面では発動しないことも合わせて固定。

## 変更していないもの

- 60 枚デッキ、LightGBM 行動ランカーとその適用範囲、Yushin Ito 1000 戦の標的ランカー
- v24/v25 の終端勝利、アーチャルドン圧力、ミストエネルギー、ハンマー、
  フェザンディピティ、退避、バックアップ構築ロジック
- v25 のノココッチ循環後山札計算（`_dudun_cycle_post_deck`）
