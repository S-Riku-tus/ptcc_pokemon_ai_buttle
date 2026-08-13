# v18変更一覧

## P0: 勝敗へ直結する修正

- `_active_route_plan()`を追加。
  - 現在のMAINで提示された合法手だけを列挙。
  - Powerful Handの最終手札、打点、KO、勝利を計算。
  - 最小手数、低い山札消費、Supporter温存の順で確定ルートを選択。
- `_boss_target_score()`へActive維持ルートとの支配比較を追加。
  - Activeで勝てるなら、勝利しないBossを禁止。
  - Activeの取得サイドが多いなら、低サイドBossを禁止。
- 改造ハンマーで効果防止エネルギーを除去できる経路を計算。
- 進化、ふしぎなアメ、リッチエネルギー、ノココッチ、Dawn、Hildaの
  確定手札増加をルートへ統合。
- 相手ベンチなしの確定KOを盤面全滅勝利として95,000点に固定。
- Activeノココッチから盤面全滅へつながる特性を88,000点に固定。

## 診断

- `active_route_plans`
- `active_route_hammer_actions`
- `active_route_draw_actions`
- `terminal_win_attack_gates`
- `terminal_pivot_abilities`
- `dudun_block_only_body`
- `dudun_block_no_bench`
- `dudun_block_deck_floor`
- `dudun_block_ability_lock`
- `dudun_declined_by_score`
- `dudun_used_for_terminal_pivot`

## 変更しなかったもの

- 60枚のデッキ
- rankerモデルと特徴量
- v17の攻撃不能・一時無敵・フリーザー・キチキギスex処理
- MLの安全なbench/evolve限定スコープ

## 撤回した試作
