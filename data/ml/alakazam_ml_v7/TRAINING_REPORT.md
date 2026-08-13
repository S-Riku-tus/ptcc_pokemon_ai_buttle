# Alakazam ML v7 candidate report

> 2026-07-18 v7.1 addendum: the 70% `attack-turn rate` gate discussed below
> used all engine turns on which the agent made any selection. A rank-1 replay
> audit measured Majkel1337 at only 35.35% by that definition despite a 79.14%
> win rate. The corrected gate uses attack-opportunity conversion and MAIN-only
> idle turns. See `ATTACK_CONTINUITY_REPORT.md` for the v7.1 retrain, exact
> feature ablation, and fresh 200-game results. Historical v7 numbers below are
> retained as the record of the original decision.

## Decision

Create `alakazam_ml_v7`, but keep its new ranker shadow-only. Restrict any
future live-override experiment to guarded bench choices. The added corpus
improved bench ranking consistently, but attack and evolution did not clear
their action-level gates.

## 1. Input audit

Sources:

- five top20 full replay ZIPs
- v5 ladder submissions `54788067` and `54795292`
- ten top21-40 Alakazam rank ZIPs containing sixteen submissions

Combined corpus:

The new rank ZIPs exposed an ingestion bug: the reader and manifest builder
previously reused the first `submission.json` for later sub-bundles in a
combined ZIP. Both layers now group by replay bundle scope. The corrected audit
recognizes all 23 submissions rather than 17.

## 2. Dataset and model

- policy scope: ACTIVE MAIN decisions with one selected legal option
- feature count: 225
- usable decisions: 320,519
- candidate rows: 3,698,955
- unresolved decisions: 6
- alignment rate: 99.9981%
- model: 50-tree LightGBM LambdaRank distilled to pure Python JSON
- model SHA256: `53427ad91da6c12db1f802441699e0123dc7902d0993a8cf8dba106b71715ce0`

The top21-40 data contributed 3,249 episodes, 3,265 trajectories, 148,845
decisions, and 1,726,889 candidate rows. Labels and leakage exclusions are
unchanged from v6.

## 3. New-teacher future holdout

Test set: the last 20% of episodes within each of the sixteen added
submissions, totaling 34,336 decisions. The baseline is the final v6 model,
which saw none of the added submissions. The candidate prediction is from the
v7 time-holdout model trained without those future episodes.

| Metric | v6 | v7 time model | Delta |
|---|---:|---:|---:|
| Top-1 | 59.928% | 60.374% | +0.446 pp |
| Top-3 | 85.409% | 84.343% | -1.066 pp |
| MRR | 0.73800 | 0.73776 | -0.00024 |
| Accepted Top-1 | 77.963% | 78.711% | +0.748 pp |
| Fallback rate | 65.057% | 64.527% | -0.530 pp |

Action Top-1 deltas:

- bench: +2.579 pp
- attack: -1.269 pp
- evolve: -0.893 pp

## 4. Fixed old-teacher holdout

Submission `54662660` remains the same 53,782-decision test set in v6 and v7.

- overall Top-1: 61.041% -> 60.102% (-0.939 pp)
- bench Top-1: 62.920% -> 65.639% (+2.719 pp)
- attack Top-1: 62.072% -> 62.019% (-0.053 pp)
- evolve Top-1: 79.033% -> 76.739% (-2.293 pp)

Bench is the only current live-scope action that improves on both independent
checks. Attack and evolution therefore returned to deterministic fallback.

## 5. Teacher-log analysis

Top21-40 corpus:

- 3,249 episodes / 3,265 target-seat trajectories
- 1,886 wins / 1,378 losses; trajectory win rate 57.76%
- 393 observed deckout losses, about 28.5% of non-winning trajectories
- losing games averaged 17.20 turns and 7.95 cards left
- winning games averaged 14.69 turns and 11.94 cards left

Losing games were longer and contained more setup actions per game, but fewer
attacks: 5.04 attacks per loss versus 5.70 per win. This supports preserving
v6's attack-continuity rules and does not support giving the regressed ML attack
ranker live authority.

Largest difficult matchup groups in these teacher logs:

- Team Rocket's Mewtwo ex: 28/105 wins (26.7%); 31 of 77 losses were deckouts
- Mega Kangaskhan ex: 200/440 wins (45.5%); 152 of 240 losses were deckouts
- Marnie's Grimmsnarl ex: 230/443 wins (51.9%)

These are observational teacher results, not causal v7 battle estimates.

## 6. Deck decision and live A/B

The added logs contain three near-identical teacher lists. The rank-2 exact
deck accounts for 1,360 trajectories at 60.22% wins. The largest one-card
variant, Enhanced Hammer 3 / Nighttime Mine 3, accounts for 1,828 trajectories
at 55.80% wins. Team strength and matchmaking confound this comparison, so it
is supporting rather than conclusive evidence.

The first v7 ablation adopted the rank-2 exact list because it is the corpus
reference and the existing fallback already contains role gates for Enriching
Energy and Shaymin:

- Dudunsparce -1
- Max Rod -1
- Enriching Energy +1
- Shaymin +1

This also removed one draw-engine copy in a corpus where deckout is a frequent
loss path. The deck change was not retained in v7.

## 8. Remaining ML improvements

1. Train and calibrate action-specific models or heads. A single global ranker
   improved bench while regressing evolution and attack.
2. Add a submission-balanced sampling ablation. Rank and outcome weights help
   Top-1, but current weights do not cap large submissions.
3. Add sequence features for attack continuity, turns since last attack, and
   optional setup after an attacker becomes ready. Current rows are mostly
   single-state snapshots.
4. Add matchup-aware public-state features for protection and stall engines.
   Team Rocket's Mewtwo and Mega Kangaskhan teacher matchups remain weak.
5. Keep reward/value learning separate. These logs currently train imitation
   labels; outcome is only a mild sample weight.
6. Treat attack continuity as attack-opportunity conversion, not the legacy
   all-engine-turn rate. The replay audit found no local attackable END choices;
   current higher priorities are boardout and mirror deckout.

## 9. Artifacts

- config: `ml/configs/alakazam_v7_candidate.json`
- processed data: `data/ml/alakazam_ml_v7_candidate/processed`
- models: `data/ml/alakazam_ml_v7_candidate/models`
- offline reports: `data/ml/alakazam_ml_v7_candidate/reports`
- replay-level analysis: `reports/top21_40_teacher_analysis.json`
- runtime: `agents/alakazam_ml_v7`
