# grimmsnarl_ex_friend_v1

友人から共有されたオーロンゲ (Grimmsnarl ex) デッキ／エージェント。
**外部由来**であり、自分の `marnies_grimmsnarl_ex_vN` 反復ラインとは別系統として扱う。

## 位置づけ

- 自分のライン (`marnies_grimmsnarl_ex_v1` 〜 `v7`) は「デッキ固定でロジックのみを反復」する系譜。
  そこに外部コードを混ぜると親子関係が追えなくなるため、独立したディレクトリに置く。
- 用途は主に次の2つ:
  1. **比較対象** — 自分の最新版と head-to-head を回して、デッキ構築・方針の違いを測る。
  2. **派生元** — 良ければ `scripts/new_agent.py` で自分のラインへコピーして改造する。

## ファイル

| ファイル | 状態 | 内容 |
| --- | --- | --- |
| `deck.csv` | ✅ 配置済み | 60枚 / 19種 |
| `main.py` | ✅ 配置済み | 約53KB |
| `policy_base.py` | なし | `agents/_base/policy_base.py` が使われる |
| `metadata.json` | ✅ 配置済み | デッキ差分・検証状況を記録 |

`agents/` に置いてよいのは Kaggle ランタイムのファイルのみ。
学習データ・レポート・joblib モデルは入れない（リポジトリ README の方針）。

## 受領後の手順

```powershell
# 1. 静的検証（60枚デッキ・構文・禁止物の混入チェック）
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent grimmsnarl_ex_friend_v1

# 2. 自分の最新版と対戦させる
.\.venv\Scripts\python.exe .\scripts\self_play.py `
  grimmsnarl_ex_friend_v1 marnies_grimmsnarl_ex_v7 --games 20
```

ベア名で解決できるので、`grimmsnarl/` を付けたパス指定は不要。

## デッキ差分（対 marnies_grimmsnarl_ex_v7）

**2枚違い。** deck hash `c20a8a46f5c63577` / v7 は `e2e03fe8ef9592b1`。

| | 枚数 |
| --- | --- |
| Handheld Fan (1161) | v7: 2 → friend: **0** |
| Pokégear 3.0 (1122) | v7: 0 → friend: **1** |
| Tool Scrapper (1137) | v7: 0 → friend: **1** |

残り58枚は完全一致。ただし2枚違う以上、対戦結果は
「ロジックのみの差」ではなく**デッキ差込みの総合差**として読む必要がある。
ロジックだけを切り分けたい場合は、どちらかの `deck.csv` を他方にコピーした
検証用ディレクトリを作って回す（過去に `*_policy_*_deck_*` 系の
run で使っている手法）。

### デッキ全体

```
10x Basic {D} Energy      4x Marnie's Impidimp      1x Pokégear 3.0
 2x Froslass              3x Marnie's Morgrem       1x Tool Scrapper
 4x Munkidori             3x Marnie's Grimmsnarl ex 4x Poké Pad
 2x Snorunt               3x Rare Candy             2x Boss's Orders
 4x Buddy-Buddy Poffin    1x Unfair Stamp           4x Team Rocket's Petrel
 3x Night Stretcher       1x Dawn                   4x Lillie's Determination
 4x Spikemuth Gym
```

## 出所メモ

受領時に以下を埋める:

- 提供者:
- 受領日:
- 元になったバージョン／実績（ラダー順位・レート等）:
- 改変の可否・共有範囲について取り決めがあれば記載:
