# Grimmsnarl ML v25 — ladder post-mortem

Date: 2026-08-15
Submission: `55507909` · reported rating 910.6 · 52 completed episodes

Reproduce:

```
python scripts/analyze_grimmsnarl_v20_ladder.py --run v25=data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909 ...
python scripts/analyze_grimmsnarl_v25_wall_engine.py
python scripts/probe_grimmsnarl_v25_wall.py
python scripts/narrate_grimmsnarl_wall_game.py --run <run> --episode <id> --seat <n>
```

## 1. Rating accounting

| run | games | win rate | opp mean | opp_mean + Elo(wr) | opp>=950 |
|---|---:|---:|---:|---:|---:|
| v22 (4 submissions) | 194 | 0.624 | 896.3 | 984.1 | 54/97 = 0.557 |
| v24 (2 submissions) | 87 | 0.655 | 797.7 | 909.2 | 3/12 = 0.250 |
| **v25** | **52** | **0.538** | **879.0** | **905.8** | **3/15 = 0.200** |
| AlphaTCG peer `55350342` | 120 | 0.542 | 1081.6 | 1110.6 | 59/113 = 0.522 |

The implied 905.8 reproduces the reported 910.6, so nothing exotic happened:
v25 was paired at v22's opponent strength and converted 8.6 pp less of it.

No agent-side failure explains it. All 52 episodes end `DONE` for our seat,
zero `ERROR`/`TIMEOUT` steps, `actTimeout` is 0 and `runTimeout` 2000 s.

## 2. The deficit is one matchup, not a general regression

Stratified by opponent-rating band x matchup class, v22's rates applied to
v25's own 52 pairings predict 0.652; v25 scored 0.538 (z = -1.72). Overall
v25 vs v22 is Fisher p = 0.27 — this run alone does not prove a regression.

Split by matchup class it stops being uniform:

| class | v22 | v25 | AlphaTCG |
|---|---:|---:|---:|
| race (everything that can be KO'd) | 0.653 (144) | **0.735 (34)** | 0.566 (76) |
| wall/tank | 0.540 (50) | **0.167 (18)** | 0.500 (44) |
| Crustle / Ogerpon / Cornerstone families | 0.545 (22) | **0.125 (8)** | 0.571 (7) |

v25 is at or above v22 on race matchups (p = 0.42 in v22's favour is not
there — the point estimate is v25's). The entire shortfall sits in the games
where the opponent's Active is damage-immune to Marnie's Grimmsnarl ex.

## 3. What actually happens in those games

Per attack, total board damage (Active + Bench splash, measured from replay HP):

| run | scope | attacks | mean damage | 0-damage share | KO/attack |
|---|---|---:|---:|---:|---:|
| v22 | vs immune Active | 56 | 7.0 | 73.2% | 0.14 |
| v25 | vs immune Active | 67 | **1.8** | **92.5%** | 0.08 |
| AlphaTCG | vs immune Active | 9 | 22.2 | 55.6% | 0.22 |

Game shape in the same matchups:

| run | games | win rate | own turns | mean total turns | deck-out losses |
|---|---:|---:|---:|---:|---:|
| v22 | 22 | 0.545 | 7.5 | 15.3 | 0 |
| v25 | 8 | 0.125 | 13.2 | 26.8 | 2 (+1 at 1 card) |
| AlphaTCG | 7 | 0.571 | 6.3 | 12.6 | 0 |

Two of the losses were from a *winning* prize position:

* `93033449` — 46 turns, we needed 1 more prize, opponent needed 4, our deck
  hit 0. Loss.
* `93030632` — 41 turns, 3 prizes to 4 in our favour, our deck hit 0. Loss.

Across 314 v22 + AlphaTCG games there is not one deck-out.

## 4. Mechanism

Both walls prevent damage **from attacks** only:

* Crustle (345): "Prevent all damage done to this Pokémon by attacks from your
  opponent's Pokémon {ex}." Grimmsnarl ex is ex → 0.
* Cornerstone Mask Ogerpon ex (117): "…by your opponent's Pokémon that have an
  Ability." Grimmsnarl ex has Punk Up → 0.

Two cards in the 60 put damage on them anyway, because neither is an attack:

* **Froslass — Freezing Shroud**: during every Pokémon Checkup, 1 damage
  counter on each Pokémon that has an Ability, except Froslass. Both walls
  have an Ability.
* **Munkidori — Adrena-Brain**: move up to 3 damage counters onto one of the
  opponent's Pokémon.

Directly observable in AlphaTCG episode `92295407` (won, same opponent deck
hash `8f6a9933` that v25 lost to twice): two Froslass in play and the benched
Crustles lose exactly 20 HP per turn — 160 → 140 → 120 → 100 — while our
Active is Froslass, not Grimmsnarl ex.

Per own turn spent facing a wall:

| run | Froslass in play | Adrena-Brain | attacks |
|---|---:|---:|---:|
| v22 | 0.89 | 0.87 | 0.44 |
| **v25** | **0.30** | 0.77 | **0.63** |
| AlphaTCG | **1.29** | 1.33 | 0.43 |

v25 keeps roughly a quarter of the teacher's Freezing Shroud clock on the
board and spends the difference swinging for zero. It is not that it cannot
build them — it evolves 5.0 Froslass per wall game — they are not kept alive
or in play at the same time.

A teacher-forced probe (`scripts/probe_grimmsnarl_v25_wall.py`, 873 walled
decisions from v25's own boards) shows v22 would have attacked the wall
*more* often (78 vs 67). So this is not a per-decision ranker regression that
a same-board probe can see; it is a trajectory difference — board composition
and game length — and the small samples on both sides (8 vs 22 games,
Fisher p = 0.09) mean the size of the effect is not yet pinned down.

## 5. Why the offline metric could not see it

From `train_alphatcg_focus3.json`:

```
hard_state_support/train/wall_recovery       = 18388
hard_state_support/validation/wall_recovery  = 0
hard_state_support/test/wall_recovery        = 0
split_submissions: train 22, validation 1, test 1   (both = AlphaTCG)
best_iteration = 228   (num_boost_round 2500)
```

Early stopping and the headline "+9.21 pp strict Top-1" were both computed on
14 AlphaTCG games containing **zero** wall decisions. The teacher itself met a
wall in only 3 of 120 games, and the 3x focus weight amplified that blind
spot. The model was selected, and the candidate promoted, on a measurement
that had no coverage of the matchup that decided the run.

228 trees out of a 2500 budget is also the failure signature recorded for
patience-200 early stopping in earlier versions.

## 6. What to change, in order

1. **Fix the promotion metric before the policy.** Model selection must not
   run on a validation block that has 0 rows in a hard-state class. Build a
   mixed validation set (AlphaTCG chronological block + a stratified
   `wall_recovery` sample from the 18,388 training rows held out), re-run with
   `--early-stopping 800`, and require non-inferiority on `wall_recovery`
   Top-1 before any submission.
2. **Deterministic wall guard.** v20/v21 shipped `wall_break.py`; v22 dropped
   it and v25 inherited the gap. Reinstate a narrow, sweep-verified guard:
   while the opponent's Active is in the immune set, forbid an attack whose
   measured board damage is 0 unless the Bench splash takes a prize, and rank
   (a) a second Froslass in play, (b) Adrena-Brain onto the wall every turn,
   (c) Boss's Orders onto a non-wall body, above it.
3. **Deck clock guard.** Past ~10 own turns with deck <= 8, stop mining the
   deck (Punk Up's 5-Energy search, surplus draw). Two games were lost from a
   winning position to our own deck; v22 and the teacher lost none in 314.
4. **Do not re-tune on this run.** 52 games cannot separate 0.538 from 0.652.
   Judge the next candidate on the >=950 cell and on matchup cells, with v22
   as a paired control in the same window.

Upside if the wall cell only reaches v22's level (0.545) at v25's exposure
(8 of 52 games): +3.4 wins, 0.538 -> 0.603, about +45 rating points. That is
the size of this defect — real, but it does not by itself reach 1000.
