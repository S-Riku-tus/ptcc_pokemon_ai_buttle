# Current Repository Status

Last updated: 2026-07-11 JST

## Overview

このリポジトリは、Kaggle `pokemon-tcg-ai-battle` 向けのエージェント、提出ログ、対戦リプレイ分析を管理する作業場です。

現状の主軸は `Alakazam / フーディン` 系デッキです。特に `alakazam741_v2` が最新の自分のデッキで、直近の検証対象になっています。以前試した Cynthia/Garchomp 系や Mega Lucario 系は、現状の勝率・方向性から外れているため archive 側に退避済みです。

## Active Agents

現在、通常の開発対象として残しているエージェントは次の通りです。

| Path | Status | Notes |
| --- | --- | --- |
| `agents/alakazam741_v1` | active | フーディン v1。v2との比較・退行確認用に保持。 |
| `agents/alakazam741_v2` | active | 現在の最新候補。submission `54523210` の検証対象。 |
| `agents/_base` | active | 共通処理。 |
| `agents/_opponents` | active | 対戦相手・参考用。 |

## Archived Agents And Runs

弱かった、または現時点の主軸から外したものは `archive/` に移動しています。

| Path | Notes |
| --- | --- |
| `archive/agents/cynthia_garchomp_v1` | Cynthia/Garchomp 系。 |
| `archive/agents/cynthia_garchomp_v2` | Cynthia/Garchomp 系。 |
| `archive/agents/mega_lucario_v1` | Mega Lucario 系。 |
| `archive/runs/20260710_182043_cynthia_garchomp_v1_sub54521273` | Cynthia/Garchomp の取得済み実行結果。 |

`archive/` は `.gitignore` 対象です。大きなログや古い試行錯誤をリポジトリ履歴に載せない運用にしています。

## Data Layout

ログ取得は、提出IDごとに `data/runs/` 以下へ run 単位で保存する運用です。

| Path | Purpose | Git |
| --- | --- | --- |
| `data/runs/` | run 名つきの取得ログ、episodes、replays、metadata、deck snapshot。 | ignored |
| `data/submissions/` | 旧方式または submission 直下保存用のデータ置き場。 | ignored |
| `data/logs/.gitkeep` | ディレクトリ保持用。 | tracked |
| `data/replays/.gitkeep` | ディレクトリ保持用。 | tracked |
| `data/summaries/.gitkeep` | ディレクトリ保持用。 | tracked |

現状、`data/` と `archive/` の中で git 追跡されている実体ファイルは `.gitkeep` のみです。対戦ログ本体・replay・zip は追跡対象外です。

## Latest Alakazam v2 Run

最新の `alakazam741_v2` 取得結果は次の run です。

| Item | Value |
| --- | --- |
| Run path | `data/runs/20260711_003324_alakazam741_v2_latest58_sub54523210` |
| Submission ID | `54523210` |
| Run name | `alakazam741_v2_latest58` |
| Deck name | `alakazam741_v2` |
| Episodes | 58 |
| Replays | 58 / 58 |
| Logs | 116 / 116 |
| Failures | 0 |

直前の 56 episode 版から新しく追加されていた 2 試合も取得済みです。

| Episode | Result | Notes |
| --- | --- | --- |
| `85224015` | loss | Mega Starmie ex 相手。序盤の盤面崩壊で `loss_no_pokemon`。 |
| `85223158` | win | Alakazam + Battle Cage 相手。`win_opp_no_pokemon`。 |

## Latest Alakazam v2 Results

58 episode 時点の大まかな結果です。

| Scope | Wins | Losses | Win Rate |
| --- | ---: | ---: | ---: |
| All | 36 | 22 | 62.1% |
| Public-like set | 35 | 22 | 61.4% |

主な敗因は次の通りです。

| Loss Reason | Count | Interpretation |
| --- | ---: | --- |
| `loss_last_prize` | 10 | 普通に取り切られる負け。テンポ・盤面維持・ミラー対策が重要。 |
| `loss_no_pokemon` | 5 | 盤面崩壊。Dragapult、Mega Starmie などに顕著。 |
| `loss_self_deckout` | 4 | 山札切れ。v2で新しく目立った問題。 |
| `loss_other` | 3 | 個別ログ確認が必要なその他負け。 |

勝ち筋は `win_last_prize` と `win_opp_no_pokemon` が中心です。つまり、フーディンの攻撃性能や盤面制圧力自体は機能しています。一方で、勝てる場面でも展開・ドロー・グッズ使用を優先しすぎて、山札や手札、攻撃タイミングを壊している可能性が高いです。

## Current Analysis

現時点の見解は、`alakazam741_v2` はデッキコンセプト自体が弱いというより、プレイ順序とリソース制御で強さを出し切れていない、というものです。

特に重要な問題は次の通りです。

1. 倒せる場面でも攻撃を最優先できていない可能性がある。
2. 通常の KO 攻撃より、展開・進化・特性・手札補充のスコアが高くなる場面がある。
3. 山札残りが少ない場面でも、任意のドロー・サーチ・手札増加行動が止まりきっていない。
4. `Alakazam 245`、`Shaymin 343`、`Psyduck 858`、`Genesect 142`、`Lucky Helmet 1156`、`Wondrous Patch 1146` など、コードには参照があるがデッキに入っていないカードが存在する。
5. `_item_locked()` が MAIN 以外の選択文脈でも過剰に反応する可能性がある。
6. スタジアム対策が薄く、Dunsparce/Dudunsparce 系の機能停止や Battle Cage 系のミラーに弱い。

## Recommended Next Steps

次は、いきなりデッキを大きく変えるより、まず `alakazam741_v2` の判断ロジックを直して `v3` として検証するのがよいです。

優先度の高い修正は次の順です。

1. `_lethal_now()` を実際の攻撃優先判定に組み込む。
2. 現在の相手アクティブを倒せるなら、原則として攻撃を展開行動より優先する。
3. 山札残り 10 枚以下では、即勝利につながらない任意ドロー・サーチをかなり強く抑制する。
4. Powerful Hand の火力を、行動前の固定評価ではなく、行動後の手札枚数変化込みで評価する。
5. `_item_locked()` を MAIN コンテキスト限定にする。
6. コードで参照している tech カードと `deck.csv` の不一致をチェックし、入っていないカード前提のロジックを無効化する。
7. Stadium 枚数と Xerosic 系の手札干渉を小さく A/B テストする。

## Validation

直近で確認した検証結果です。

```text
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent alakazam741_v2
Validation passed
deck_size: 60
unique_cards: 20
warnings: 0
```

```text
.\.venv\Scripts\python.exe -m pytest
4 passed
```

## Git And Ignore Status

`archive/`、`data/runs/`、`data/submissions/` は `.gitignore` 対象です。ログ・replay・zip が増えても git 履歴が重くならないようにしています。

`git status` 実行時に、ユーザーのグローバル git ignore である `C:\Users\shiba\.config\git\ignore` に対する permission warning が出ることがあります。これはこのリポジトリ内の変更ではなく、現状の作業内容とは直接関係しません。

## Operational Rule

今後は次の運用がよいです。

1. デッキや判断ロジックを変えたら、新しい agent ディレクトリまたは明確な run name を使う。
2. Kaggle submission ID ごとに `scripts/fetch_submission_logs.py --run-name ... --deck-name ... --deck-dir ...` で取得する。
3. 取得結果は `data/runs/` に残すが、git には載せない。
4. 弱かった試行は `archive/` に移動する。
5. 強い候補だけを `agents/` に残す。

