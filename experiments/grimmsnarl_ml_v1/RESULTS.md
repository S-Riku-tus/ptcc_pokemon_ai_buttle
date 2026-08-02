# Grimmsnarl ML v1 — top-50 imitation

Date: 2026-08-02
Target: reproduce top-50 Marnie's Grimmsnarl ex pilots' actions at ≥90%
Result: **91.13% end-to-end runtime agreement** on held-out games of the
pinned pilot (n=485, Wilson 95% 88.27–93.35%).

## Why this corpus is better than the Alakazam one

`data/kaggle_grimmsnarl_top50` holds 4,772 replays from 25 top-50 teams. 21 of
them run byte-identical decks (hash `9714ab5c3996f6cc`), giving 3,655 usable
trajectories and **148,028 MAIN decisions** — 18% more than the entire
Alakazam corpus, and from 21 pilots rather than one.

Deck differences are trivial across the four observed variants. Our previous
`marnies_grimmsnarl_ex_v7` list is itself a top-50 list (`21c3f20980827cb2`,
2 teams); it differs from the dominant list by three slots only:
−2 Handheld Fan, +1 Pokégear 3.0, +1 Tool Scrapper. v1 adopts the dominant
list.

## Are the 21 pilots one policy?

`scripts/analyze_grimmsnarl_teacher_corpus.py` reduces every MAIN decision to
a signature and compares choices when two decisions share one. At the `menu`
level (turn bucket + offered option multiset, the only level that collides
often enough to measure):

| corpus | self-agreement | cross-team agreement | gap |
|---|---:|---:|---:|
| dominant deck, 30 games/team | 86.18% | 85.11% | 1.1 |
| dominant deck, all games | 81.74% | 77.45% | 4.3 |
| all four deck variants | 81.06% | 75.54% | 5.5 |

Close enough to pool, but not one policy. The per-pilot breakdown shows why:

**Agreement with the field is inversely related to rating.** The rank-4 pilot
(1220.2) agrees with the field 74.35% of the time; the rank-39 pilot (1062.3)
agrees 82.07%. The field consensus *is* the ~1060-rated policy. What separates
the strongest pilots is exactly the part a consensus model averages away.

## Models

All numbers are strict Top-1 on decisions never used for fitting or
configuration selection. Early stopping is on Top-1 — LightGBM's built-in
metrics are disabled, because `lgb.early_stopping` halts on whichever tracked
metric stalls first and NDCG would otherwise pick the tree count for a model
deployed on argmax (the v33 defect).

| model | validation | test | order-insensitive |
|---|---:|---:|---:|
| pooled field, global split | 0.7515 | 0.7416 | 0.9342 |
| **+ pilot as a categorical feature** | 0.7921 | **0.7731** | 0.9410 |
| + tuned (lr 0.03, 383 leaves) | 0.7931 | 0.7750 | 0.9413 |
| pilot-conditioned, per-pilot split | 0.8025 | **0.7968** | 0.9442 |
| single-pilot model, 16556346 | 0.8720 | 0.8218 | 0.9497 |
| single-pilot model, 16371703 | 0.6708 | 0.6732 | 0.9136 |

Conditioning on the acting pilot is worth +3.2 points pooled. It also beats
dedicated single-pilot models on both pilots tested (86.6% vs 82.2% for
16556346; 69.9% vs 67.3% for 16371703): the 21-pilot corpus supplies shared
mechanics that a 5k-decision single-pilot corpus cannot.

## Per-pilot agreement (pilot-conditioned model, per-pilot split)

| rank | rating | team | n | Top-1 | 95% CI |
|---:|---:|---|---:|---:|---|
| 48 | 1048.5 | 16421840 | 485 | **0.9155** | 0.887–0.937 |
| 36 | 1064.7 | 16462035 | 1350 | 0.8926 | 0.875–0.908 |
| 30 | 1077.6 | 16494330 | 758 | 0.8839 | 0.859–0.905 |
| 24 | 1086.7 | 16556346 | 812 | 0.8682 | 0.843–0.890 |
| 17 | 1101.8 | 16452116 | 1379 | 0.7796 | 0.757–0.801 |
| 5 | 1172.6 | 16422241 | 545 | 0.7450 | 0.707–0.780 |
| 4 | 1220.2 | 16371703 | 1363 | 0.7029 | 0.678–0.727 |

The same inverse relationship. Ranking by rating and ranking by imitability
point in opposite directions.

## Runtime parity

The Alakazam line lost 5.16 points between what its ranker chose and what its
agent played, because a safety shell overrode MAIN decisions. v1 has no shell
over MAIN. `scripts/evaluate_grimmsnarl_v1_runtime.py` replays the held-out
games through the shipped `main.py`:

| pilot | offline | runtime | delta |
|---|---:|---:|---:|
| 16421840 (shipped) | 0.9155 | **0.9113** | −0.42 |
| 16494330 | 0.8839 | 0.8813 | −0.26 |
| 16556346 | 0.8664 | 0.8721 | +0.57 |
| 16385817 | 0.8433 | 0.8445 | +0.12 |

0 feature errors, 0 scoring errors, 0 illegal indices. Latency p95 65 ms per
MAIN decision.

## Where the remaining 9% goes

On the pooled test block: 79.7% correct, 14.7% same-turn ordering errors (the
model played something the teacher also played that turn, in another order),
0.7% same-action-type divergence, 4.9% genuine divergence. Order-insensitive
turn-set agreement is 94.4%.

This is the same shape the Alakazam v36 report identified, and it bounds what
further ranking work can buy: even a perfect ordering fix leaves ~5.6%.

## Self-play

20 games on the cg engine vs `marnies_grimmsnarl_ex_v7`: **16-4**, 0 crashes,
0 illegal selects, ~10 ms/move. This measures the mirror only and is not a
rating estimate.

## Open items

1. No ladder rating yet. Per `docs/` history and the rating-noise finding,
   compare versions by opponent-bucketed win rate, not by a single rating.
2. The pinned pilot is a one-integer change (`--teacher-team` at export).
   16421840 maximises fidelity (91.1%, teacher 1048.5); 16494330 trades 3
   points of fidelity for 29 points of teacher rating. Both are worth a
   ladder run.
3. Non-MAIN contexts (~47 per game, more than MAIN) are still rule-driven and
   were never measured against the teachers. That is the largest unexplored
   block of behaviour.

## Artifacts

- `teacher_corpus_dominant.json` / `teacher_corpus_all.json` — homogeneity
- `corpus_v1_report.json` — 148,028-decision extraction
- `train_all_teams.json`, `train_team_conditioned.json`,
  `train_cond_tuned.json`, `train_cond_perteam.json`, `train_single_*.json`
- `runtime_*.json` — end-to-end agreement and latency
