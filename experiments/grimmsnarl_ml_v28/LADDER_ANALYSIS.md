# v28 ladder analysis: the first post-v22 version that is not a regression

Submission `55526859`, 35 rated games + 1 validation self-match, fetched
2026-08-15 into `data/runs/grimmsnarl/20260815_grimmsnarl_ml_v28_sub55526859`.
Live rating 968.7, public rank 175 of 6827 at the 2026-08-15 23:16 JST board
snapshot; 975.7 after the last stored game.

Compared against v22 (`55479857`, `55483874`, `55486680`, `55486691`), v24
(`55496021`, `55496665`), v25 (`55507909`, `55517142`), v26 (`55520389`) and
v27 (`55521760`) — 480 stored games in total.

Reproduce:

```
python scripts/fetch_submission_logs.py --submission 55526859 \
    --output data/runs/grimmsnarl/20260815_grimmsnarl_ml_v28_sub55526859 \
    --deck-dir agents/grimmsnarl/grimmsnarl_ml_v28
python scripts/fetch_kaggle_top100_snapshot.py --top-n 60
python scripts/build_grimmsnarl_version_games.py \
    --output experiments/grimmsnarl_ml_v28/version_games.csv
python scripts/probe_grimmsnarl_v28_footprint.py \
    --run data/runs/grimmsnarl/20260815_grimmsnarl_ml_v28_sub55526859 \
    --submission 55526859 --agent agents/grimmsnarl/grimmsnarl_ml_v28 \
    --output experiments/grimmsnarl_ml_v28/footprint_v28_run.json
python scripts/analyze_grimmsnarl_v28_ladder.py
python scripts/analyze_grimmsnarl_v28_ladder.py --reference v25 \
    --output experiments/grimmsnarl_ml_v28/ladder_verdict_vs_v25.json
python scripts/analyze_grimmsnarl_v28_games.py
python scripts/analyze_grimmsnarl_v28_field.py
python scripts/analyze_grimmsnarl_v28_levers.py
python scripts/analyze_grimmsnarl_v28_adrena_lever.py
python scripts/analyze_grimmsnarl_v28_band_mix.py
```

Logs and JSON for every table are in this directory.

## 1. The change bound, unlike the last three versions

Teacher-forced replay of all 36 stored episodes, reading `main._LAST_TRACE` so
that the v22 ranker's answer and v28's final answer come from the same pass
over the same history:

| | v28 | v27 | v26 |
|---|---|---|---|
| single-pick decisions evaluated | 2898 | 2755 | 3051 |
| reproduces its own played action | 2898 (100%) | 99.93% | 99.97% |
| v22 ranker reproduces the played action | 2255 (77.8%) | 99.64% | 99.71% |
| **decisions differing from the v22 ranker** | **487 (16.8%)** | **8 (0.29%)** | **8 (0.26%)** |
| normalised per 35 games | 473 | 8 | 7 |

v26 and v27 were v22 with eight decisions changed. v28 is a different policy:
302 ordinary changes, 183 mirror changes, 2 in the wall cell.

The wall switch itself is nearly inert by construction: only 98 of 2898
decisions ran under the v22 wall ranker (96 wall-Active, 2 wall-public), and
in that cell v28 *is* v22, so it produced 2 differences. **v28 is the v25 race
ranker in 96.6% of its decisions**, plus deterministic guards. That matters
for section 3, because v25 measured −141 Elo.

Runtime: 12.8 s of the 600 s bank per game, 0 component load errors, 0
non-standard step statuses.

## 2. What the 968.7 is worth

A Kaggle rating converges to `mean(opponent rating) + 400·log10(w/(1−w))`.

| run | n | record | win rate | opp mean | implied strength |
|---|---|---|---|---|---|
| v22 | 190 | 120-70 | 0.632 [0.561,0.697] | 915.2 | **1008.8** |
| v24 | 85 | 56-29 | 0.659 | 816.4 | 930.7 |
| v25 | 85 | 49-36 | 0.576 | 824.7 | 878.2 |
| v26 | 40 | 28-12 | 0.700 | 706.6 | 853.8 |
| v27 | 34 | 21-13 | 0.618 | 789.0 | 872.3 |
| **v28** | **35** | **24-11** | **0.686 [0.520,0.815]** | **860.5** | **996.0** |

Controlled logistic on outcome with opponent rating and turn order held fixed:

| version dummy | vs the v22 pool | vs the v25 pool |
|---|---|---|
| v24 | −72.7 Elo (p 0.178) | +68.0 (p 0.258) |
| v25 | −141.4 Elo (p 0.0089) | — |
| v26 | −118.4 (p 0.146) | −31.6 (p 0.706) |
| v27 | −132.7 (p 0.0866) | −23.3 (p 0.772) |
| **v28** | **−5.2 (p 0.94)** | **+135.3 (p 0.093)** |

v28 is the first post-v22 version that is statistically indistinguishable from
the champion, and it is ~135 Elo above the v25 policy whose ranker it runs.
The most likely reconciliation is that v25's deficit was concentrated in cells
v28 changed (wall 0.300 → 0.500, deck-outs 3 → 0, long grind games 11.3 own
turns → 5.75 in the wall cell), but with n=35 this is a point estimate, not a
proof.

Calendar control — the field drifts about 100 Elo a day, so only same-day rows
compare. On 2026-08-15: v25 828.3, v26 853.8, v27 872.3, **v28 996.0**;
pooled non-v28 strength on that day 850.5 over 110 games.

Not converged: first half 0.765, second half 0.611, last 10 games 0.600. Only
7 games came against 950+ opponents (3-4).

Clean run: 0 gate violations, 0 board-outs, 0 deck-outs — the only version in
the table with zeros in all three.

## 3. Where v28 is good

* **The mirror.** 9-2 (0.818) against our own 60. Both seats of every mirror
  were walked, so the peer comparison is same deck, same game:

  | per own turn | us | same-deck peer | Δ |
  |---|---|---|---|
  | Adrena-Brain | 1.136 | 0.714 | +0.421 |
  | attacks | 0.678 | 0.589 | +0.089 |
  | Shadow Bullet | 0.627 | 0.554 | +0.074 |
  | decisions / game | 98.5 | 79.3 | +19.2 |
  | first Shadow, own turn | 2.90 | 3.00 | −0.10 |

  Against the four mirror peers rated **above** us at pairing we went 4-0 with
  first Shadow on own turn 2.75 against their 3.75, 5.75 bodies left against
  2.75, and 1.25 prizes left against 4.50.

* **Tempo.** Best first-Shadow own turn of any version (2.71 vs v22's 3.01)
  and best first attack (shared turn 4.20 vs 4.78).

* **The second seat.** 12-6 at opponent mean 887 (strength 1007.5). The
  v26/v27 second-seat collapse (704.8 / 759.9) is gone.

* **Race matchups.** Mega Lucario 3-0, Archaludon 2-0, Alakazam 4-2,
  Dragapult 1-1; non-wall cell 22-9 (0.710, strength 1015.9).

* **Highest early-ability rate of any version** — 1.20 Adrena-Brain uses in
  own turns 1-3 per game, against 1.00-1.11 for v22-v27.

## 4. Where v28 is bad

**Every loss is a blowout.** In v28's 11 losses we hold 4.18 prizes on average
while the opponent holds 1.27; 8 of 11 losses left 4+ of our prizes unclaimed
and only 1 was within 2 prizes. There is no close-game conversion lever here —
the games are decided before the endgame.

**Three archetypes carry the whole deficit, and they concentrate at the top.**

| family | all versions | v28 | median current score of those opponents | best current rank |
|---|---|---|---|---|
| Ogerpon | 5-22 (0.185) | 2-2 | 913.3 | 130 |
| Mega Lopunny / Froslass | 15-18 (0.455) | 1-2 | 959.8 | 58 |
| other: Hydrapple ex | 4-9 (0.308) | 0-1 | 1070.4 | 41 |
| everything else | 281-126 (0.690) | 21-6 | — | — |

Their share of the field rises with opponent strength: 7.0% below 900, 15.8%
at 900-950, **31.6% at 950+**. Inside the 950+ band we are 0.286 against those
three and 0.560 against everything else. **Ogerpon at 950+ is 0-13.**

Lifting those three cells to a coin flip is worth +19.9 Elo pooled and
**+47.1 Elo inside the 950+ band**, which is the band the rating converges on.

Mechanism, from the two v28 Ogerpon losses (`93307572`, `93320220` — identical
shapes: 8 turns, 4 own turns, 3 attacks, 0 Adrena-Brain, 5 bodies left, we
take 2 prizes and they take 4): Teal Mask Ogerpon ex is Grass, Marnie's
Grimmsnarl ex is Grass-weak, so our 320 HP attacker dies in one hit and pays 2
prizes each time. The wall ranker cannot reach this — it fires on
Crustle/Cornerstone-style Actives, and only 98 decisions in the whole run ran
under it. Tempo cannot reach it either: our first Shadow in Ogerpon *wins*
averages own turn 2.20, the fastest of any family.

Mega Lopunny / Froslass is different: first Shadow is 2.93 in wins and 2.88 in
losses, so tempo does not discriminate at all, and the opponents in that cell
are uniformly strong (many 1000+). It is the highest-rated family we meet.

## 5. The lever table

Every flag fitted against winning over all 480 games with opponent rating and
turn order held fixed.

| state | with | without | controlled Elo | p |
|---|---|---|---|---|
| bodies left ≥ 5 | 0.834 | 0.162 | +584 | 0.000 |
| Adrena-Brain per own turn ≥ 1.0 | 0.794 | 0.479 | +291 | 0.000 |
| 5+ Adrena-Brain | 0.766 | 0.449 | +290 | 0.000 |
| zero Adrena-Brain | 0.302 | 0.668 | −315 | 0.000 |
| 3+ Adrena-Brain in own turns 1-3 | 0.840 | 0.612 | +229 | 0.001 |
| any Adrena-Brain in own turns 1-3 | 0.720 | 0.510 | +178 | 0.000 |
| …same, only games reaching own turn 5+ | 0.723 | 0.530 | +160 | 0.000 |
| Munkidori energised by own turn 3 | 0.647 | 0.523 | +149 | 0.014 |
| Rare Candy used | 0.675 | 0.536 | +109 | 0.004 |
| first Shadow by own turn 2 | 0.722 | 0.610 | +104 | 0.006 |
| 2+ Lillie's Determination | 0.663 | 0.585 | +98 | 0.008 |
| Froslass evolved | 0.631 | 0.642 | −17 | 0.621 |
| gate violation | 0.645 | 0.635 | −17 | 0.805 |

"Bodies left" is an outcome, not a lever. The Adrena-Brain family survives
both the per-turn denominator and the restriction to games that reached own
turn 5, so it is not only "we won fast".

**But it is mostly not ours to choose.** Adrena-Brain needs a {D} Energy on
Munkidori *and* a damage counter on one of our own Pokémon. Of the 194 games
with no use in own turns 1-3:

| blocked by | games | win rate |
|---|---|---|
| no damage on our own board | 120 (61.9%) | 0.525 |
| no Munkidori in play | 27 (13.9%) | 0.481 |
| no {D} Energy on Munkidori | 17 (8.8%) | 0.588 |
| **all preconditions met, not used** | **30 (15.5%)** | **0.433** |
| used it | 286 | 0.720 |

So the addressable slice is 30 games in 480 (6.3%). Lifting them from 0.433 to
0.720 is worth about +8 Elo pooled — real, cheap, and much smaller than the
archetype gap. Caveat: availability here is inferred from board state, not
from the option list, so before building anything the next probe must count
decisions where the Adrena-Brain option was actually offered and declined.

## 6. Against the board

Snapshot 2026-08-15 23:16 JST, 6827 teams: rank 1 = 1225.1, rank 10 = 1190.3,
rank 60 = 1056.9, rank 100 = 1012.4, rank 150 = 982.0, **rank 175 = us at
968.7**, rank 200 = 960.0.

Distance: +44 to the top 100, +88 to the top 60, +222 to the top 10.

Of v28's 35 opponents, 22 are still active on the board. Against teams
currently ranked ≤500 we are 9-2, ≤200 3-1, ≤100 1-1.

Both of our two submission slots are live: v28 at 968.7 and v27, still playing
at 859.6 as of 14:24 UTC. The public leaderboard takes the max of a team's
slots, so the second slot is currently contributing nothing.

## 7. What follows

1. **Free, no modelling**: replace the v27 slot with a second run of v28. The
   board takes the max of two slots and two runs of the same agent differ by
   pairing luck alone.
2. **The only lever that changes the rating** is the Ogerpon / Lopunny /
   Hydrapple block, which is 32% of the 950+ band and worth +47 Elo there. It
   is not a tempo problem and not a wall-ranker problem: against Ogerpon we
   already attack first and still go 0-13 above 950. It is a prize-trade
   problem — a Grass attacker one-shots a 320 HP, 2-prize Grimmsnarl ex.
   Any fix has to change what we present to that attacker, not how fast we
   present it.
3. **Cheap and testable**: the 30 games where Adrena-Brain was available in
   the first three own turns and was not used. Probe the option list first.
4. Do not read v28's Ogerpon 2-2 as a fix. Both wins came against ~850-rated
   pilots; the prior lineage was 2-5 in that same sub-900 band, and 0-13 above
   950.
