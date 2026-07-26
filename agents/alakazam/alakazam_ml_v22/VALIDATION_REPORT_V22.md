# v22 Validation Report

## 実施日

2026-07-26

## 静的・単体検証

- pytest: 121 / 121 成功
- v22固有Golden-state: 12 / 12 成功
- `scripts/validate_agent.py`: 成功
- deck: 60枚
- unique card IDs: 22
- warning: 0
- fallback policy例外: 0
- observation fallback: 0

## v20継承確認

| ファイル | v20との関係 |
|---|---|
| `deck.csv` | SHA-256一致 |
| `ranker_model.json` | SHA-256一致 |
| `main.py` | SHA-256一致 |
| `ml_runtime.py` | SHA-256一致 |
| `ml_features.py` | SHA-256一致 |
| `policy_base.py` | SHA-256一致 |
| `fallback_policy.py` | v20から242行追加、7行変更/削除 |

`fallback_policy.py`以外の実行時構成は最新版v20と同一である。

## 最新v20ログ再生

- source: `20260725_v20_run2_sub54976903`
- decisions: 3,046
- v22 policy成功: 3,046
- policy exception: 0
- semantic changes from v20: 113
- semantic agreement with v20: 96.29%
- model override: 12
- live model rate: 1.51%

狙った変更の観測数:

- continuity Dudunsparce choices: 9
- sufficient-chain draw blocks: 57
- backup pre-fuel choices: 43
- first Dunsparce rebuild choices: 34
- attack-route promotions: 113
- shield promotions: 65

この集計は同じ実観測に両方策を当てたteacher-forced監査であり、v22の仮想勝率
ではない。前の行動を変えた後の盤面推移や相手の応答は再現しない。

## 守られた重要不変条件

- 場のフーディンへの同ターン攻撃給エネが後続先貼りより高い
- 最後の1体でノココッチを使わない
- 確定KOを手札消費で壊さない
- ミスト/完全無効への0ダメージ攻撃を強制しない
- ふしぎなアメ即KOをユンゲラー経由より優先できる
- きぜつ後は攻撃可能な経路を盾より優先
- 攻撃経路がなければノコッチ/ノココッチを進化ラインより優先
- 通常時キチキギスexへ攻撃用エネルギーを段階投資しない

## 未検証

- 新規提出の実戦Rating
- Rating 900/1000での安定性
- 変更後の対戦系列を含む反実仮想勝率
- 公式`cg`を含む提出tar.gz（ローカルには互換shimしかない）

Rating目標は設計目標であり、達成を保証する検証結果ではない。新しい実戦ログを
取得後、Archaludon、Grimmsnarl、攻撃4回未満、後続先貼りの成否を再監査する。
