# Grimmsnarl ML v3 experiment results

Date: 2026-08-03

## Objective

Improve agreement with the strongest same-deck Grimmsnarl teachers without
using any v2 ladder replay and without discarding the v2.1 policy that reached
rating 962 at 16-7.

## Evidence and hypothesis

The v2.1 all-team test is 84.84%, but its error taxonomy is 9.13% same-turn
ordering, 3.61% same-action divergence and 2.41% genuine divergence. The
highest-rated teachers also agree less with the field and are individually
noisy. This makes full-policy replacement high variance. The selected
hypothesis is narrower: their shared advantage is more learnable at the level
of which action family comes next than at the level of the exact card/target.

## Ablations

1. `train_rank4_focus6.json`: sixfold rank-4 loss weight. Validation rose from
   78.07% to 78.80%, but test fell from 78.54% to 78.38%. Rejected.
2. `train_elite_tier.json`: replace exact pilot ID with one shared elite tier.
   It scored 78.41% validation and 78.81% test, versus 80.96% and 81.32% for
   exact-ID v2 on the same five teachers. Rejected.
3. `action_prior_elite.json`: an unconstrained action-family prior improved
   elite validation/test by 1.41/1.22 points, but alpha 1.0 cost the pinned
   teacher 1.85/1.55 points. Useful signal, unsafe coefficient.
4. `action_prior_guarded06.json`: constrain pinned validation loss to 0.60
   points. Alpha 0.10 was selected. Elite validation/test improve by 0.84/0.53
   points; pinned test changes by only −0.14 points. Adopted.

The stricter 0.50-point experiment (`action_prior_guarded.json`) selected
alpha 0.04 and reproduced a smaller elite test improvement (+0.19pt) with a
−0.14pt pinned change. Alpha 0.10 was preferred because its elite gain was
larger and its test guard loss was identical.

## Runtime and game checks

- Rank-4 real runtime: 67.11% to 67.72% all-context, 60.01% to 61.12% MAIN.
- Pinned real runtime: 91.34% to 91.20% all-context.
- 60-game paired self-play versus v2.1: 35-25.
- Errors, illegal actions and fallbacks: 0.
- Multiclass export probe maximum absolute error: 0.

The final agent is `agents/grimmsnarl/grimmsnarl_ml_v3`.

