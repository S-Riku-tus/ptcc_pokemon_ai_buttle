# Grimmsnarl ML v3 validation

## Why v3 is two-stage

The v2.1 chronological test error is dominated by same-turn ordering: 9.13%
of all decisions select an action that the teacher also performs later in the
same turn. Exact teacher-ID conditioning reproduces the current rank-30 pilot
well (91.96%), but the highest-rated pilots are less deterministic and much
harder to imitate. Rank 4 is only 78.54% even when its own ID is supplied.

Two direct specialisation attempts were rejected:

| Candidate | Validation | Test | Decision |
|---|---:|---:|---|
| rank-4 loss weight 6x | 78.80% | 78.38% | test did not improve |
| shared elite tier | 78.41% | 78.81% | −2.55/−2.51pt vs exact-ID v2 |

The adopted model shares only action-family order across the elite cohort and
leaves concrete candidate selection to v2.1.

## Guarded coefficient selection

The action prior was trained on 62,594 chronological training decisions from
teams 16371703, 16422241, 16463316, 16561259 and 16531269. It has 501 public
state inputs plus 55 symmetric legal-menu/v2-score summaries. Multiclass
log-loss stopped at 143 boosting rounds.

An unconstrained coefficient of 1.0 improved elite test agreement by 1.22
points, but reduced pinned-teacher agreement by 1.55 points. v3 therefore
uses a validation guard: maximise elite strict Top-1 while pinned-teacher
validation loss remains at most 0.60 points.

| Split | v2 pinned policy | v3 alpha 0.10 | Delta |
|---|---:|---:|---:|
| elite validation | 75.98% | 76.82% | +0.84pt |
| pinned validation | 90.45% | 89.94% | −0.51pt |
| elite test | 75.84% | 76.37% | +0.53pt |
| pinned test | 91.96% | 91.82% | −0.14pt |

All five elite test pilots improve individually: +0.66, +0.09, +0.70,
+0.58 and +0.18 points in rank order 4, 5, 9, 13 and 11. Elite MAIN improves
from 66.83% to 67.85%; contexts outside MAIN are unchanged at this coefficient.

## Shipped-runtime replay parity

The real `main.py` was evaluated teacher-forced on the chronological replay
block, so JSON model conversion, semantic duplicate collapse and intra-turn
history are included.

| Teacher | Scope | v2.1 | v3 | Delta |
|---|---|---:|---:|---:|
| rank 4 / 1220.2 | all contexts | 67.11% | 67.72% | +0.61pt |
| rank 4 / 1220.2 | MAIN | 60.01% | 61.12% | +1.11pt |
| pinned rank 30 | all contexts | 91.34% | 91.20% | −0.14pt |
| pinned rank 30 | MAIN | 88.92% | 88.65% | −0.27pt |

There were zero feature, base-score and action-prior errors. Native LightGBM
raw multiclass scores and the exported JSON runtime agree exactly on fixed
conversion probes (maximum absolute error 0).

## Paired self-play

With seed reset per game and alternating seats, v3 beat v2.1 35-25 in 60
games (58.3%). Both sides had zero errors, illegal selections and policy or
observation fallbacks. Average time was 20.98 ms/move for v3 versus 17.44
ms/move for v2 in this run. Replay evaluation measured about 34 ms mean and
66 ms p95 for v3.

This is a regression test, not a rating estimate. The decisive next evidence
is a separate ladder submission plus opponent- and matchup-bucketed logs.

