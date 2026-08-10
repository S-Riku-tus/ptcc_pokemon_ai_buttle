# Grimmsnarl ML v14: v13 ladder autopsy and guarded residual design

## Decision

v14 keeps the exact v8 champion ranker in every matchup. It removes both
full-policy specialists introduced by v13 and allows only two narrow residuals:

1. the v10 Petrel search correction for a dead Unfair Stamp, backed by the
   3,642-game / 21-pilot rating gradient; and
2. a new wall safety gate. When v8 selects Shadow Bullet into an Active that
   takes zero damage, Bench-30 takes no prize, and fallback still has a
   non-closing development action, that one fallback action replaces the
   attack. The free swing remains legal when fallback also attacks or ends.

The 60 cards, v8 model and proof-based narrow planner are unchanged. The v9
model is not packaged. The public router is telemetry only and reports zero
policy switches.

## What the v13 logs establish

The attached analysis is directionally correct, but the same-board replay adds
an important effect-size measurement.

| measurement | v13 result |
| --- | ---: |
| rated games | 116 |
| record | 66-50 (56.9%) |
| default route | 30-16 |
| mirror route | 27-17 |
| Alakazam route | 8-7 |
| wall route | 1-10 |
| v13-v8 differences on 28 specialist games | **514 / 2,450 (21.0%)** |

Wall did not behave as a validator. Depending on the game it replaced 20-58
decisions, including setup, search targets, attachments and target selection.
That is a whole policy, and its 1-10 result is consistent with the observed
post-first-attack idle problem.

Alakazam is less decisive but still does not justify a full replacement. Across
all 18 Alakazam archetype games v13 went 11-7. The 15 games that publicly locked
to v9 went 8-7; the three late-detection games that remained v8 went 3-0, with
opponents rated 915, 1,014 and 1,040. On identical logged states, v9 differed
from v8 on 8.36% of decisions in wins and 15.96% in losses. This is descriptive,
not a causal estimate, but it supplies no confidence gate that would make the
generic v9 model a safe Alakazam expert.

Therefore v14 does not pretend that a matchup label turns a generic consensus
model into a trained specialist. A genuine expert remains future work and must
be trained only on matchup-specific trajectories and promoted on closed-loop
matchup results.

## Counterfactual footprint

`v14_vs_v8_specialists_same_board.json` teacher-forces both agents through the
same 28 v13 specialist games.

| policy | differences from v8 | rate | changed contexts |
| --- | ---: | ---: | --- |
| v13 | 514 / 2,450 | 20.98% | 9 contexts |
| v14 | **10 / 2,450** | **0.41%** | context 7 and MAIN only |

The ten v14 differences are exactly five Petrel residuals and five wall
development gates. v14 removes 98.1% of v13's specialist-induced changes while
retaining an explicit correction for the concrete wall failure mode.

## Verification

- 196 v14 tests pass.
- Static validation passes: 60 cards, 19 unique IDs, no warnings, unchanged
  deck hash `9714ab5c3996f6cc`.
- Clean extracted archive imports with all four load-error fields null and
  returns the 60-card deck.
- 18 local legality games produced zero crashes and zero illegal selections:
  v14-v8 4-2, v14-Alakazam-v35 5-1, v14-first-policy-on-Crustle-deck 6-0.
  These tiny, unseeded-engine samples are runtime checks, not power estimates.
- On the 28 specialist replay states, the wall gate fires five times total and
  never more than twice in one game; v13 changed 20-58 decisions in wall games.

## Submission artifact

`agents/grimmsnarl/grimmsnarl_ml_v14/grimmsnarl_ml_v14.tar.gz`

- size: 10,932,697 bytes
- SHA-256: `44236229EB9EFB9F47CA5BBB7A485960A4C9DC47F9E9192C256D9C7543F1137D`
- archive entries: 21

## Ladder interpretation and next gate

v14 is a causal challenger, not a claim of 1,100 strength. Submit two replicas
and aggregate them. The first useful audit is not peak rating but:

- at least 100 rated games combined after placement;
- Wall route record and post-first-attack idle below 2 turns/game;
- Alakazam and mirror split by turn order and opponent rating;
- mean rating delta at own rating >=950 and >=1,000;
- exact count of Petrel and wall-guard firings;
- zero crash / illegal / timeout.

Only after v14 supplies a stable champion trajectory should a learned v15
expert be trained. Its unit of prediction should be a multi-turn prize macro
route (Froslass spread, Munkidori prize, Boss KO, backup attacker, wall unlock),
not merely the next action. Promotion must require the specialist to beat v14
closed-loop in its matchup while leaving default and mirror untouched.
