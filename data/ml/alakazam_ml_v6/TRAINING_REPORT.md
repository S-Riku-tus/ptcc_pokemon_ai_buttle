# Alakazam ML v6 candidate training report

## Decision

The two v5 submissions were successfully added to the training corpus and a
new model was produced. The model is installed in `agents/alakazam_ml_v6` for
shadow scoring only. Live override remains disabled because the evidence is
mixed: overall ranking improved slightly on the added-replay holdout, but the
Top-1 accuracy of attack, bench, and evolution decisions declined.

## What the model learns

This is supervised behavior cloning with a LightGBM LambdaRank model. It is not
reinforcement learning and it does not estimate the long-term value of an
action.

One training decision is an `ACTIVE` main-selection observation with at least
two legal options. Each legal option becomes one candidate row. The option that
the same replay seat selected at the next step receives label `1`; all other
legal options receive label `0`. The model learns to rank the selected option
above the alternatives within that decision.

The 225 policy features cover:

- Turn number, action count, first-player seat, and per-turn usage flags.
- Both players' observable hand count, deck count, prize count, Bench count,
  total HP, total Energy, and status counts.
- Active Pokemon identity, HP, damage, Energy, Tool count, and appearance flag.
- Bench capacity and aggregate HP, damage, Energy, and low-HP counts.
- Counts of 21 important card IDs in the acting player's hand, field, and
  discard, plus the opponent's field.
- Abra-Kadabra-Alakazam route readiness, attacker Energy readiness, and the
  Dunsparce-Dudunsparce engine state.
- Candidate action type, card, target, area, HP, Energy, hand cost, and attack.
- Powerful Hand damage, estimated KO/overkill, and whether a candidate preserves
  or breaks an available KO.
- Action-state interactions for Hammer, Xerosic, Boss, Retreat, Energy,
  Evolution, Bench, Ability, Trainer, and End Turn.

The policy features exclude future observations, reward/winner, deck hash,
leaderboard rank, and opponent private hand card IDs. Rank, deck distance,
outcome, seat confidence, alignment confidence, and rare-action balancing are
used only to set sample weights. A win has weight factor `1.05`, a loss `0.95`,
and a result without a win/loss signal `0.90` before normalization and clipping.

## Added data

| Submission | Replays | Wins | Losses | Decisions |
| --- | ---: | ---: | ---: | ---: |
| 54788067 | 57 | 34 | 23 | 2,036 |
| 54795292 | 35 | 18 | 17 | 1,399 |
| Total | 92 | 52 | 40 | 3,435 |

All 92 replays were usable, their target seats were identified with confidence
`1.0`, and no duplicate trajectory was removed. They added 48,118 candidate
rows and a third deck cluster.

## Corpus totals

| Metric | v5 corpus | Augmented corpus | Delta |
| --- | ---: | ---: | ---: |
| ZIPs / submissions | 5 | 7 | +2 |
| Replay files | 3,158 | 3,250 | +92 |
| Expert trajectories | 3,161 | 3,253 | +92 |
| Decisions | 168,239 | 171,674 | +3,435 |
| Candidate rows | 1,923,948 | 1,972,066 | +48,118 |
| Teams | 5 | 6 | +1 |
| Deck clusters | 2 | 3 | +1 |

The final alignment rate is `99.997%` with five unresolved decisions and no
excluded replay trajectory.

## Offline comparison

The fixed team, submission, and deck holdouts showed small mixed changes:

| Holdout | Top-1 delta | MRR delta | Accepted Top-1 delta |
| --- | ---: | ---: | ---: |
| Team | +0.57 pp | +0.34 pp | +0.15 pp |
| Submission | -0.30 pp | -0.20 pp | +3.18 pp |
| Deck | -0.41 pp | -0.17 pp | -0.02 pp |

For the most direct test, the last 20% of episodes from submissions `54788067`
and `54795292` were held out. An old-corpus model was compared with a model
trained on the old corpus plus the first 80% of the new episodes:

| Metric | Old corpus | Augmented | Delta |
| --- | ---: | ---: | ---: |
| Top-1 | 44.39% | 44.65% | +0.26 pp |
| Top-3 | 75.33% | 77.55% | +2.22 pp |
| MRR | 0.6216 | 0.6283 | +0.0067 |
| Attack Top-1 | 62.65% | 61.45% | -1.20 pp |
| Bench Top-1 | 62.11% | 60.00% | -2.11 pp |
| Evolution Top-1 | 47.34% | 46.75% | -0.59 pp |

This supports retaining the model as a shadow advisor, not promoting it to live
control. The new model SHA-256 is
`4c3e6fefd1164ed3148ed936e2e08a784c0173d3d6d1571845fdbc08f03a7ecf`.

## Runtime verification

- Agent validation: passed; 60-card deck; no warnings.
- Focused tests: 24 passed.
- 20-game v6-v5 shadow smoke: v6 13-7, crashes 0, illegal actions 0,
  timeouts 0. This smoke validates runtime safety only; it is not model
  promotion evidence because shadow mode returns the deterministic action.

## Artifacts

- Configuration: `ml/configs/alakazam_v6_candidate.json`
- Processed dataset: `data/ml/alakazam_ml_v6_candidate/processed`
- Models: `data/ml/alakazam_ml_v6_candidate/models`
- Pipeline reports: `data/ml/alakazam_ml_v6_candidate/reports`
- Added-replay comparison:
  `data/ml/alakazam_ml_v6_candidate/reports/added_v5_holdout_comparison.json`
- Smoke report:
  `data/runs/ml_v6_evaluation/post_retrain_smoke/20260718_144804_alakazam_ml_v5_vs_alakazam_ml_v6`
