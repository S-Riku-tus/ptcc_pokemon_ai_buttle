# v6 ladder run, submission 55290882 (fetched 2026-08-06, rating 996.6)

55 episodes, all `COMPLETED`, one of them a non-public validation episode that is
excluded from every win rate below, leaving 54 rated games. Read against the
gates in `RESULTS.md` §8, which were written before the run.

Sources: `data/runs/grimmsnarl/20260806_grimmsnarl_ml_v6_sub55290882`,
`behaviour_v6_ladder_sub55290882.json` (teacher-forced probe on the run's own
boards), `../v6_meta_gap_analysis/runs_banded.json`.

The probe reproduces v6's own play on its own boards at **0.9818** agreement,
against the instrument's measured self-agreement floor of 0.9811, so the
behaviour numbers below are the deployed behaviour rather than a reconstruction.

## 1. Pre-registered gates

| # | gate | target | measured | verdict |
| --- | --- | --- | --- | --- |
| 1 | Froslass evolve per offered own turn | 60-85% | **73.2%** (41/56 played; 40/56 agent) | pass |
| 2 | Froslass evolve on net-negative-shroud decisions | <= 55% | **41.9%** (13/31) | pass |
| 3a | `escalation_offered` / scored decisions | 1.5-4.0% | **2.77%** (135/4866) | pass |
| 3b | `escalation_moved` / `escalation_offered` | 10-30% | **36.3%** (49/135) | **miss, high** |
| 4 | Grimmsnarl ex evolve per turn | ~62.3% | 64.9% (72/111) | pass |
| 4 | Dark attachment per turn | ~72.8% | 77.6% (208/268) | pass, better |
| 4 | enabling attachment | ~95.4% | 97.1% (167/172) | pass |
| 4 | Boss's Orders per turn | ~35.6% | 39.3% (35/89) | pass |
| 4 | Adrena-Brain damaged-Grimmsnarl pass | ~35.9% | 35.7% (85/238) | pass |
| 4 | counters moved to maximum | 100% | 100% (375/375) | pass |
| 4 | best-prize Bench-30 kill | 100% | 97.4% (37/38) | marginal miss |
| 5 | Darkness left in deck, overall / from turn 5 | >= 4.0 / >= 2.4 | 4.09 / 2.56 | pass |
| 6 | crashes, illegal selects, timeouts | 0 | 0 | pass |

Gate 6 detail: 55/55 episodes `COMPLETED` with a final `DONE` status for our
seat, no non-standard per-step status, and the probe reports `feature_errors` 0,
`score_errors` 0, `planner.errors` 0.

**The headline gate passed and it is the first behaviour change in this line that
visibly landed on the live ladder.** v4, v4.5 and v5 all evolved Froslass on
100% of offered own turns (47/47, 56/56, 59/59). v6 played it on 73.2%, inside
the pre-registered 60-85% band and inside the current top pilots' range
(16561259 0.726, 16422241 0.846, 16452116 0.873).

**Gate 3b missed on the high side.** The class fires as predicted (2.77% against
a predicted 2.79% live and a 1.52% teacher-forced probe) but changes the answer
on 36.3% of firings against a predicted 17-19%. The behavioural consequence is
still inside its own band, so this is a mis-set expectation rather than a
different mechanism; the probe measured `moved` on v5's boards, and on v6's own
boards the class fires on later, more contested turns.

**The two marginal items are not divergences.** In the one Bench-30 case that was
not the best-prize kill, the replay and the teacher-forced agent chose the same
option, so it is deployed behaviour rather than a probe disagreement. Per own
turn, the 12 turns whose shroud ledger was net negative all still ended with
Froslass in play (12/12) even though only 41.9% of the individual net-negative
*decisions* took the evolve: v6 refuses the evolve while the ledger is negative
and takes it later in the same turn once the ledger flips. With 12 such turns
this is not worth reading further, but it is not what "reduce the evolve on bad
boards" was expected to look like.

## 2. Outcome, banded by opponent rating

A final rating cannot rank two versions in this repo (~40-84 point spreads on
identical code), and a pooled win rate tracks the pool, so all four runs are
reported the same way. Elo-style performance rating is
`mean opponent + 400 log10(W/L)`, an approximation - Kaggle's system is not
Elo-400.

| run | games | W-L | win rate | mean opponent | < 900 | 900-1000 | 900+ pooled | rating | perf rating |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4 | 51 | 32-19 | 0.627 | 967.1 | 5/7 | 17/26 | 27/44 = 0.614 | 1031.2 | 1057.7 |
| v4.5 | 67 | 47-20 | 0.702 | 843.0 | 24/32 | 21/31 | 23/35 = 0.657 | 979.2 | 991.4 |
| v5 | 65 | 43-22 | 0.661 | 859.6 | 27/35 = 0.771 | 16/28 = 0.571 | 16/30 = 0.533 | 963.7 | 976.0 |
| **v6** | 54 | **38-16** | **0.704** | 865.6 | 19/23 = 0.826 | 18/29 = 0.621 | 19/31 = **0.613** | **996.6** | 1015.9 |

v6 beats v5 on five aligned signals at a nearly identical opponent pool (865.6
vs 859.6): overall win rate, the sub-900 band, the 900-1000 band, the pooled
900+ band, and the final rating. **None of it is significant**: Fisher exact
gives p = 0.695 overall, p = 0.609 on the 900+ band, p = 0.746 under 900. By
the standard in [[kaggle-ladder-rating-noise]] the agreement of several weak
signals is the promotion evidence, and that standard is met for v6 over v5.

v4 remains level with v6 on the 900+ band (0.614 vs 0.613) from a much stronger
pool, and still holds the highest rating of the four.

Seat split: v6 0.750 first (24/32) and 0.652 second (15/23), against v5's 0.686
and 0.645. The going-second gap that defined the v3 analysis is not visible in
either run.

## 3. Per-archetype, and what it does not say

| opponent archetype | v5 | v6 |
| --- | --- | --- |
| Marnie's Grimmsnarl ex (mirror) | 16/21 = 0.762 | 15/22 = 0.682 |
| Alakazam | 6/13 = 0.462 | 7/10 = 0.700 |
| Mega Kangaskhan ex | 4/8 | 5/6 |
| Cinderace | 6/7 | 4/4 |
| Mega Lucario ex | 2/3 | 3/3 |
| Mega Lopunny ex | - | 2/2 |
| Dragapult ex | - | **0/2** |
| Teal Mask Ogerpon ex | 1/2 | 0/1 |

§8 said not to read the Alakazam matchup below ~40 games, and 10 games is far
below it; the same applies to the mirror's apparent 8-point drop. The two cells
worth noting for their absence are the ones that matter at 1100+: Dragapult ex
(ranks 1 and 2 of the current leaderboard) is 0/2, and the Mega Lopunny ex +
Mega Froslass ex list that holds 5 of the current top 20 appears once. See
`../v6_meta_gap_analysis/RESULTS.md`.

## 4. What this run changes about the plan

1. **v6 replaces v5 as champion.** The behaviour change landed, every unchanged
   behaviour held or improved, and five outcome signals agree in its favour.
2. **The Petrel / Unfair Stamp class is now the clearest decision-level target.**
   Taking an Unfair Stamp that cannot be played this turn went the wrong way on
   this run: **87.9%** (29/33) against v5's 75.0% and the current top pilots'
   0.330-0.495. The class is already implemented in `ml_runtime.py` and held
   behind `GRIMMSNARL_ESCALATION_CLASSES`, and it is the one gap whose rating
   gradient is significant (rho -0.626, p = 0.0029). n = 33, so the run does not
   establish a regression - it establishes that the gap is still open.
3. **The Dark attachment gap is closed.** 78.6% taken when offered (v5 73.6%),
   inside the top pilots' 0.760-0.847, and 58.7% per own turn against v5's
   55.7% and the 57% target. Attachment should come off the target list.
4. **Nothing here touches the matchup deficit.** The two archetype families that
   carry the whole shortfall against the current top 40 are still unmeasured on
   our side: 3 games total across both.
