# Current Repository Status

Last updated: 2026-07-13 JST (v5/v6最終レート確認後)

## Overview

このリポジトリは、Kaggle `pokemon-tcg-ai-battle` 向けのエージェント、提出ログ、対戦リプレイ分析を管理する作業場です。

現状の主軸は `Alakazam / フーディン` 系デッキです。特に `alakazam741_v2` が最新の自分のデッキで、直近の検証対象になっています。以前試した Cynthia/Garchomp 系や Mega Lucario 系は、現状の勝率・方向性から外れているため archive 側に退避済みです。

最終レートのメモは `docs/ALAKAZAM_RATING_LOG.md` に分離して残しています。v1=845.7、v2=855.4、v3=850.3、v4=724.8、v5=763.5、v6=700.2、v8=675.7 です。v4は実ラダー上では明確な退行、v5はそこからの部分回復ですが、まだ v1-v3 の強度帯には戻っていません。v6とv8はさらに低く、方針は変わりません。したがって、v5以降も v2 または v3 を骨格にしつつ、v4/v5/v6/v8 ログで確認できた修正点を選択的に移植する方針です。

## Active Agents

現在、通常の開発対象として残しているエージェントは次の通りです。

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

次のアクション: v3をsubmitして実ラダーで山札切れ負け0とミラー勝率改善を確認する。分析の全文は `docs/alakazam741_v3_analysis.md`。

## v3 Ladder Result And v4 (2026-07-11)

v3のsubmission `54557078` の結果 (51戦 30勝21敗 58.8%) を全数分析し、`alakazam741_v4` を作成済み。

次のアクション: v4をsubmit (`kaggle/create_submission_from_git.py` はv4に変更済み) し、
Lucario/Cinderace対面の改善を同方式で再分析する。分析の全文は `docs/alakazam741_v4_analysis.md`。

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
