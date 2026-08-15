# v27 ladder analysis, and what it is worth against v22 and v24

Submission `55521760`, final 853.3, 34 rated games + 1 validation self-match,
fetched 2026-08-15 into
`data/runs/grimmsnarl/20260815_grimmsnarl_ml_v27_sub55521760`.
Compared against v22 (`55479857`, `55483874`, `55486680`, `55486691`) and
v24 (`55496021`, `55496665`), with v25 and v26 carried along as controls.

Reproduce with:

```
python scripts/build_grimmsnarl_version_games.py
python scripts/analyze_grimmsnarl_v27_vs_champions.py
python scripts/analyze_grimmsnarl_field_freshness.py
python scripts/analyze_grimmsnarl_field_inflation.py
python scripts/analyze_grimmsnarl_field_speed.py
python scripts/analyze_grimmsnarl_rating_trajectory.py
python scripts/analyze_grimmsnarl_same_policy_pool.py
python scripts/analyze_grimmsnarl_second_seat_clock.py
python scripts/analyze_grimmsnarl_decision_coverage.py
python scripts/probe_grimmsnarl_v27_ladder_footprint.py            # v27, H2 off
python scripts/probe_grimmsnarl_v27_ladder_footprint.py --h2       # v27, H2 on
python scripts/probe_grimmsnarl_v27_ladder_footprint.py \
    --run data/runs/grimmsnarl/20260815_grimmsnarl_ml_v26_sub55520389 \
    --submission 55520389 --agent agents/grimmsnarl/grimmsnarl_ml_v26 \
    --output experiments/grimmsnarl_ml_v27/ladder_footprint_v26.json
```

Logs and JSON for every table below are in this directory.

## 0. Correction to the premise

v24 was not the most stable high-rating version. Ordered by final rating the
runs are v22 1020.4 / 1018.6 / 1000.6 / 952.8, then v24 928.1 / 911.3, then
v25 910.7 / 808.4, v26 835.5, v27 853.3. **v22 is the champion and v24 is
already 80-100 points below it**, which matches
`experiments/grimmsnarl_ml_v24/LADDER_VERDICT_2026-08-14.md`: v24's only
change, the mirror Froslass veto, measured -72.7 Elo and was shown to be
non-causal.

## 1. v27 is v22

Teacher-forced replay of all 35 stored v27 episodes, reading `main._LAST_TRACE`
so v22's answer (the `planner` index) and v27's final answer come from the same
pass over the same history:

| | v27 | v26 |
|---|---|---|
| single-pick decisions evaluated | 2755 | 3051 |
| v27/v26 reproduces the played action | 2753 (99.93%) | 3050 (99.97%) |
| v22 baseline reproduces the played action | 2745 (99.64%) | 3042 (99.71%) |
| **decisions where the version differs from v22** | **8 (0.29%)** | **8 (0.26%)** |
| games containing any difference | 4, record 3-1 | 3, record 2-1 |
| ordinary (non-wall, non-mirror) differences | 0 / 1648 | 0 / 2204 |

v27's 8 differences are 6 mirror-Froslass vetoes and 2 wall-trajectory
preservations. v26's are 7 wall-break and 1 wall-trajectory.

**The belief search contributed nothing.** With H2/H3 fully enabled against
the real engine on v27's own games: `considered` 301, `searched` 23,
`branches` 336 (204 H2, 132 H3, 66 H3 worlds) — and **0 overrides**. The
searches that completed were rejected by `not_dominant` (20), `h3_small_gain`
(14), `h3_not_dominant` (5), `h3_prize_safety` (3), `small_gain` (1); 41 were
stopped by the budget. The gate (`skip_non_mirror` 1985, `skip_non_main` 311,
`skip_already_searched_turn` 149, `skip_early` 85) is narrow enough that the
module cannot reach the game. Runtime cost is real but small: 14.9 s of a
600 s bank per game against v22's 12.5 s (Mann-Whitney p = 0.003); the
180 s per-episode search budget was essentially untouched, and no episode
finished with less than 30 s of bank left.

So **v22, v26 and v27 are one policy separated by 16 decisions across 264
games.** Everything that differs between their results is the field, the
pairing draw or chance.

## 2. Where the 853 comes from

A Kaggle simulation rating converges to `mean(opponent rating) + 400·log10(w/(1-w))`.

| run | n | record | win rate | opp mean | implied strength | final |
|---|---|---|---|---|---|---|
| v22 pooled | 190 | 120-70 | 0.632 | 915.2 | 1008.8 | 952.8-1020.4 |
| v24 pooled | 85 | 56-29 | 0.659 | 816.4 | 930.7 | 911.3, 928.1 |
| v25 pooled | 85 | 49-36 | 0.576 | 824.7 | 878.2 | 808.4, 910.7 |
| v26 | 40 | 28-12 | 0.700 | 706.6 | 853.8 | 835.5 |
| **v27** | **34** | **21-13** | **0.618** | **789.0** | **872.3** | **853.3** |

v27's raw win rate is within a point of v22's (0.618 vs 0.632, Fisher
p = 1.00). The 150-point rating difference is the 126-point difference in the
pairing draw.

Rating after the same number of games, so a 34-game run is not compared with a
57-game one:

| after 34 games | v22_a | v22_b | v22_c | v22_d | v24_a | v24_b | v25_b | v26 | v27 |
|---|---|---|---|---|---|---|---|---|---|
| rating | 965.2 | 1008.3 | 1048.6 | 930.1 | 915.7 | 888.6 | 808.4 | 799.9 | **853.3** |
| opp mean so far | 799.0 | 880.1 | 987.2 | 904.0 | 787.8 | 827.5 | 717.4 | 685.3 | **789.0** |

The two runs with a matched draw are v24_a (opp 787.8 → 915.7) and v22_a
(opp 799.0 → 965.2). Against those v27 is **-62 and -112 Elo at equal
exposure**. The byte-identical noise floor on this competition is ~77 Elo
([[kaggle-ladder-rating-noise]]), and the per-game increment has already
decayed to ±9 by game 34, so the run was near its own equilibrium — it was not
simply cut short.

## 3. The field moved, and that is most of the -62/-112

Three independent measurements, all with our policy held constant.

**(a) The same rating band got harder.** Win rate against opponents rated
700-900, per run, in submission order:

```
v22_a 0.750  v22_b 0.667  v23 0.667  v22_c 0.750  v22_d 0.700   (08-13)
v24_a 0.810  v24_b 0.773  v25_a 0.722                            (08-14)
v25_b 0.524  v26 0.619  v27 0.538                                (08-15)
```

First 24 h 66-22 (0.750) against after 24 h 51-35 (0.593), Fisher p = 0.036.
Note v25_a and v25_b are the same code ten hours apart: 0.722 → 0.524.
In a logistic fit over all 445 games, `hours_per_day` is **-100.6 Elo/day
(p = 0.0016)**, and when it is included the post-v22 lineage dummy collapses
from -115.1 (p = 0.005) to **+15.2 (p = 0.87)**.

**(b) The field started attacking earlier.** Opponent's first attack, in the
opponent's own turn ordinal — a property of their deck, not ours:

| window | games | opponent first attack | our first Shadow Bullet |
|---|---|---|---|
| 08-13 | 199 | 2.33 | 3.01 |
| 08-14 | 136 | 2.28 | 2.97 |
| 08-15 | 110 | **1.98** | 3.16 |

Mann-Whitney one-sided p = 0.001. Per version: v22 2.33, v24 2.20, v25 2.20,
v26 2.03, v27 2.06.

**(c) The second seat collapsed, with identical code.** On the pooled
v22+v26+v27 corpus:

| | going first | going second | diff | Fisher p |
|---|---|---|---|---|
| 08-13 (v22) | 62-38 (0.620) | 58-32 (0.644) | -0.024 | 0.76 |
| 08-15 (v26+v27) | 35-6 (0.854) | 14-19 (0.424) | **+0.429** | **0.0002** |

This is not an opponent-strength artefact: v22 shows no turn-order split in
*any* band (0-700 -0.111, 700-800 +0.300, 800-900 -0.090, 900-1000 -0.077,
1000+ 0.000), and restricting both windows to opponents under 900 keeps the
result (08-13 -0.055 p = 0.75, 08-15 +0.398 p = 0.0007). Dropping the mirror
and Ogerpon, whose share moves with the meta, keeps it too (+0.431,
p = 0.0007). The logistic interaction `first × is_v26_v27` is **+310 Elo,
p = 0.004**; `first × hours` is +117.6/day, p = 0.037.

Turn order is a coin flip inside the episode, so within a window the contrast
is randomised. The interaction with the window is not randomised, and 74 games
on one morning is a real caveat — but it agrees with (b), which is measured on
the opponents' own actions.

## 4. Everything that is *not* the explanation

Checked and null:

* **Runtime.** No ERROR/TIMEOUT step in any episode of any version. The
  largest single-game spend was 30.1 s of the 600 s overage bank.
* **Behaviour drift.** Per-game counts of attacks (3.88 vs v22 3.99), Shadow
  Bullets (3.41 vs 3.59), Grimmsnarl evolutions (2.06 vs 2.03), Rare Candy
  (1.09 vs 0.88), Adrena-Brain (5.62 vs 5.76), Unfair Stamp (1.06 vs 1.19),
  Boss (0.53 vs 0.76), first ready turn (2.88 vs 2.95). Section 1 already
  explains why: the actions are the same actions.
* **The v15 attack-access gate.** 3 violations in 34 games, unchanged.
* **Deck-out.** 0 in v27 and v26 (v25 had 3).
* **Matchup composition.** Stratifying v22's rates onto v27's own family ×
  turn-order exposure predicts 21.99 wins against 21 observed (z = -0.41).
  By opponent band alone the residual is -3.53 (z = -1.42). Neither is
  significant.
* **The mirror.** v27 3-3 over 6 real mirrors; v22 26-20 over 46.

## 5. What the run does say about v27 specifically

* The **restored v24 mirror-Froslass veto fired 6 times in 6 mirror games**
  (25 Froslass options offered, 6 selected by v22, all 6 vetoed). Four of
  those six were in episode `93247540`, which we lost 0-5 prizes. That is one
  game and proves nothing, but the lever was already measured at -72.7 Elo in
  v24 and shown to be non-causal; carrying it forward is unbacked risk.
* The **belief search is dead weight as gated** (section 1): 1103 lines,
  a value model, +2.4 s per game, 0 decisions changed.
* The **wall guards** changed 2 decisions in v27 and 8 in v26, in games that
  went 3-1 and 2-1. Not evaluable.

Net: v27 shipped three interventions and the ladder measured none of them.

## 6. Where the remaining points actually are

* **The ≥950 band is what sets the rating and v27 never entered it** (0 games
  at ≥950, 2 at ≥900). Pooled v22+v26+v27 we are 54-44 = 0.551 [0.453, 0.646]
  there; matching the two same-deck top-30 pilots needs ~0.616
  ([[grimmsnarl-rating-is-the-950-band]]).
* **The take side is saturated.** Over the 264-game pool, Shadow Bullet
  offered-but-not-taken is 0.19/game in second-seat wins and 0.04/game in
  second-seat losses. Offers per game are 4.19 in wins against 2.53 in losses.
  The deficit is entirely offer-side, as in [[grimmsnarl-gap-is-offer-side]].
* **6.3% of our real choices never reach any model.** Of 9861 non-forced
  decisions over 119 games, the ranker owns 9237. The 624 it does not own are
  concentrated in exactly two places: context 22 (Punk Up's multi-pick energy
  placement, 201 of 222 decisions, 1.9 per game) and context 5 (deck search,
  108 of 154). Both are decided by `fallback_policy`, both directly determine
  whether Grimmsnarl ex is online on own turn 2, and no metric scores either
  ([[multipick-selects-bypass-the-ranker]]).
* **Ogerpon is still unanswerable**: 1-16 pooled over every version, 0-3 in
  v27 ([[grimmsnarl-ogerpon-structural-counter]]).

## 7. Measurement protocol

The instrument is the problem. Over 47 hours the same code produced implied
strengths of 1041, 1031, 1012, 968 (v22, 08-13) and 862 (v26+v27, 08-15).
A change worth less than ~100 Elo is invisible to a single sequential run.

* Never compare a challenger to a champion run from a different day. Run the
  challenger and a v22 control in **both slots at the same time**
  ([[kaggle-new-submission-truncates-old]]).
* Report every version as pooled win rate against ≥950 opponents with a Wilson
  interval, plus the teacher-forced footprint (how many decisions it actually
  changes). v27's footprint — 8 of 2755 — should have been measured *before*
  spending a ladder slot: it predicts a null before the run starts.
* Any new module must state the number of ladder decisions it expects to
  change. If that number is under ~50 per 35 games, the ladder cannot see it.
