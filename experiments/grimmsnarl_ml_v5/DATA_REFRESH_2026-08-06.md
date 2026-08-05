# v5 data refresh decision — 2026-08-06

## Decision

Promote the refreshed ranker at **2,000 trees**, retaining teacher pin
**16494330**. The v5 rule policy, planner, deck, and routed-context gate remain
unchanged. This isolates the effect of new training data and avoids combining
it with an unproven pin change.

## Frozen data selection

- 4,097 games, 322,975 decisions, 21 teams, and 21 selected submissions.
- At most the newest 300 valid same-deck games per team, across submission
  versions.
- New team 16422241 / submission 55177269: rank 3, score 1146.7, 300 games.
- Manifest SHA256:
  `22f13bad91bc3bc3466d20c8a5469649de6572ca70afab2ec49a4b563a249df7`.
- No rating, recency-loss, or win/loss weights were applied. Previous rating
  and win-weight experiments reduced held-out Top-1.

## Iteration selection on the refreshed test

All rows below compare the candidate and `ranker_v4.txt` on the same 38,753
chronological test decisions. The interval is an episode-cluster bootstrap of
candidate minus baseline Top-1.

| Trees | Candidate | v4 | Delta | Delta 95% |
|---:|---:|---:|---:|---:|
| 1,238 | 0.8478 | 0.8465 | +0.0013 | [-0.0012, 0.0037] |
| **2,000** | **0.8500** | **0.8465** | **+0.0036** | **[0.0010, 0.0059]** |
| 2,500 | 0.8506 | 0.8465 | +0.0041 | [0.0016, 0.0063] |
| 3,515 | 0.8516 | 0.8465 | +0.0052 | [0.0028, 0.0074] |

The full 3,515-tree optimum exported to 79.2 MB and measured about 88 ms/move
in the initial two-game probe. The 2,000-tree export is 45.1 MB and retained a
clear refreshed-test gain. The extra 500 trees to 2,500 bought only 0.0005
Top-1, so 2,000 is the selected accuracy/runtime Pareto point.

## Cross-checks at 2,000 trees

| Benchmark | Decisions | Candidate | v4 | Delta | Delta 95% |
|---|---:|---:|---:|---:|---:|
| Frozen v4 test | 34,611 | 0.8523 | 0.8503 | +0.0020 | [-0.0008, 0.0048] |
| New rank-3 team 16422241 | 2,749 | 0.8127 | 0.7850 | +0.0276 | [0.0201, 0.0356] |
| Existing pin 16494330 | 1,587 | 0.9282 | 0.9137 | +0.0145 | [0.0007, 0.0270] |

The old frozen distribution shows no detected regression. The intended new
top-team adaptation and the existing deployed-pin fidelity both improve.

## Pin experiment

- Pin 16494330 candidate vs old v5, seed 1705: 23-17.
- Pin 16422241 candidate vs old v5, seed 1705: 23-17.
- Pin 16422241 vs pin 16494330, seed 2718: 20-20.

The higher-rated new pin has no measured local outcome advantage. Keep the
existing pin; treat a pin change as a separate future experiment.

## Promotion gate

- Each pin candidate: 144/144 inherited tests passed.
- Promoted candidate vs previous v5: 61-39 over 100 games using two seeds,
  alternating seats; Wilson 95% = [0.5120, 0.6998].
- Crashes: 0. Illegal selects: 0.
- Weighted local timing: 38.64 ms/move candidate vs 22.03 ms/move old v5
  (1.75x). This is an accepted cost but must be watched on the ladder runtime.
- Ladder performance is not measured yet.

## Artifacts

- `data/ml/grimmsnarl/processed/corpus_v5_data_refresh_candidate.npz`
- `data/ml/grimmsnarl/models/ranker_v5_data_refresh_base.txt`
- `experiments/grimmsnarl_ml_v5/data_refresh_selection.csv`
- `experiments/grimmsnarl_ml_v5/train_v5_data_refresh_base.json`
- `experiments/grimmsnarl_ml_v5/eval_*_vs_v4.json`
- `experiments/grimmsnarl_ml_v5/arena_*json`
