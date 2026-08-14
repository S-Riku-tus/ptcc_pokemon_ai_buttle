# v22 at 194 games, and what it says about v23

## Corpus

Four byte-identical v22 submissions and one truncated v23 run, refetched
2026-08-14 to completeness (remote episode count == local, every replay and
both agent logs present, zero manifest errors):

| run | sub | games | record | win rate | final rating |
|---|---|---:|---:|---:|---:|
| v22_a | 55479857 | 46 | 33-13 | 0.717 | 1000.6 |
| v22_b | 55483874 | 39 | 27-12 | 0.692 | 1018.6 |
| v22_c | 55486680 | 58 | 31-27 | 0.534 | 1020.4 |
| v22_d | 55486691 | 51 | 30-21 | 0.588 | 952.8 |
| **v22 pooled** | | **194** | **121-73** | **0.624** | |
| v23 | 55485982 | 12 | 7-5 | 0.583 | 827.4 |

v23's run was truncated: its last episode and v22_b's share the timestamp
`15:35:46`, and the two replacement submissions began at `15:36:01` and
`15:36:38`. Its 827.4 is a starting-point artifact, not a score - the same-code
v22 runs read 901.5 / 900.7 / 1144.3 / 943.3 at exactly 12 games.

Version labels were verified rather than assumed: teacher-forcing each
candidate through the stored boards, the version that played reproduces it
100.0% and its neighbour 93-96%
(`version_probe_*.json`). The 194-game footprint walk confirms this at scale -
v22 reproduced **17,376 of 17,376** stored decisions, so `base_infidelity` is 0
and every number below is measured against a walker known to be faithful.

## 1. The noise floor, measured from the inside

Fitting `won ~ opponent_rating/400 + went_first + run` over the four
byte-identical runs gives a run term of up to **-76.8 Elo (p = 0.34)** for a
difference that is exactly zero by construction. That independently reproduces
the -75.5 Elo measured on the v11 twin submission with a different sample and a
different week's pool.

We are rank 60 of the public top 60 at 1025.9. The distance to the top-20 cut
(1100) is **+74 Elo - the same size as the noise floor.** No single 50-game run
can resolve the improvement we are trying to make.

The opponent-rating term is -329 Elo per 400 rating points (p = 0.0025),
statistically consistent with the -400 a correctly calibrated Elo ladder would
produce, so the pairing pool is behaving normally.

## 2. What is now closed

* **Going second.** first 0.606 (n=104) vs second 0.644 (n=90); controlled
  `went_first` = -24.2 Elo, p = 0.66. The 69.7/50.6 split that drove v10-v15 is
  gone.
* **Alakazam.** 30-11 = **0.732** over 41 games, our best matchup with n >= 15.
  It is a surplus, not a defect.
* **Attack access (the v15 gate).** 10 violations in 194 games, and those games
  went 9-1. Not a live constraint.
* **Munkidori setup.** Benched on own turn 1.30 in wins vs 1.37 in losses;
  energised 1.61 vs 1.57 (t = 0.25). Saturated.

## 3. Measured nulls - do not spend a version on these

Controlled for opponent rating and turn order, n = 194:

| lever | Elo | p |
|---|---:|---:|
| own_first_shadow_turn | -24.8 | 0.34 |
| own_first_ready_turn | -34.5 | 0.19 |
| Unfair Stamp count | +20.1 | 0.56 |
| Boss count | +27.8 | 0.45 |
| Rare Candy count | +46.4 | 0.21 |
| Grimmsnarl ex evolutions | +47.3 | 0.18 |

The board-by-own-turn-2 gradient that justified v19/v20 has collapsed: 0.696
(n=69) vs 0.584 (n=125) raw, not significant once controlled. It is not the
0.689-vs-0.000 cliff recorded earlier.

## 4. Adrena-Brain is a scoreboard, not a lever

It is the largest raw discriminator in the corpus - 6.84 uses in wins vs 3.99
in losses (t = 6.18), and 8.37 vs 4.12 in the mirror (t = 7.41). It is also
not actionable, and the offer/take split says why:

| | wins | losses |
|---|---:|---:|
| offers per game | 14.78 | 9.07 (t = 5.75) |
| uptake when offered | 0.4625 | 0.4396 |

We decline at the same rate whether we win or lose; we are simply offered it
less when the board is worse, and Munkidori's arrival turn is identical either
way. Optimising the count directly would repeat the Powerful Hand mistake.
The same caution applies to `shadow_attacks` and `attacks`.

## 5. The one live finding: Froslass in the mirror

| mirror (n=56) | games | win rate |
|---|---:|---:|
| no Froslass evolved | 32 | **0.719** |
| one or more | 24 | **0.292** |

* controlled for opponent rating and turn order: **-366.8 Elo, z = -3.04,
  p = 0.0023**; Fisher exact on the 2x2 p = 0.0026;
* it replicates in **all four** independent submissions (0.75/0.00, 0.88/0.40,
  0.75/0.25, 0.50/0.50);
* **placebo is clean**: outside the mirror, 0.630 (n=27) vs 0.667 (n=111) -
  no effect, so this is not a generic "Froslass is bad" artifact;
* it is not just a losing board reaching for an engine: prize state at the
  moment of evolution was behind 14 / even 22 / ahead 17, and early (own turn
  <= 2) games go 3/8 while late ones go 4/16 - both bad;
* mechanism is consistent with a tempo cost, not a shroud trade: mirror games
  with Froslass show 2.21 vs 2.50 Grimmsnarl evolutions, 3.96 vs 4.25 Shadow
  Bullets and first Shadow on own turn 3.00 vs 2.78.

With ~20 levers tested, a Bonferroni-corrected p is 0.046 - it survives, but
only just, and it is observational.

### The obvious gate is backwards

v22 already computes `shroud_net`. Gating on it would make things worse:

| mirror | games | win rate |
|---|---:|---:|
| every evolution at net > 0 | 10 | **0.100** |
| at least one at net <= 0 | 14 | 0.429 |
| none | 32 | 0.719 |

A "require `shroud_net > 0`" rule refuses 26 of the 53 mirror evolutions -
precisely the ones in the games that did better. The Freezing Shroud ledger is
close to symmetric in the mirror (our 2.83 targets vs their 3.26), so it is not
the mechanism. Any v24 experiment here has to suppress the line, not condition
it on the ledger.

One tidy mechanism was checked and rejected: `shroud_net` is read off the board
*before* the evolution, so it would be off by one if Froslass were itself a
Freezing Shroud target. It is not - `FROSLASS_ID` is absent from
`ABILITY_POKEMON_IDS` - so the ledger is computed correctly and still points
the wrong way. The remaining candidate mechanism is tempo: two bench slots that
can neither attack nor retreat cheaply, and the search and Rare Candy spent
reaching them.

## 6. v23's exposure is aimed at the wrong matchup

Teacher-forced over all 194 v22 games (17,376 decisions):

| | changed | share | per game |
|---|---:|---:|---:|
| overall | 1,094 | 6.30% | 5.64 |
| games v22 **won** | 725 | 6.34% | 5.99 |
| games v22 **lost** | 369 | 6.20% | 5.05 |
| **Alakazam** (wr 0.732) | 356 | **10.79%** | 8.68 |
| **mirror** (wr 0.536) | 280 | **5.27%** | 5.00 |

v23 is a real policy change - it touches 190 of 194 games, so it is not another
inert-guard release. But it intervenes twice as hard in our best matchup as in
our worst, and slightly more in games we already won than in games we lost. Its
premise came from v22_a's 44 games, where Alakazam read 8/14; over 41 games it
reads 30/11.

## 7. Field

Of the current top 40, 3 play this exact 60-card list and 2 more play another
Grimmsnarl list - **12.5%**. The mirror is 29% of the games we are actually
paired into at ~1020, so the mirror fix is worth more now than it will be after
a climb.

## Recommendations

1. **Do not ship v23 unchanged.** Re-target it or hold it. Its footprint is
   concentrated on the matchup that is now our strongest.
2. **v24 = suppress the Froslass line in the mirror**, not a `shroud_net` gate.
   Measure the binding count offline over all 56 mirror games before spending a
   ladder slot; v18 shipped two guards that bound zero times.
   Ceiling if the association is causal: 24 games x 0.43 = ~10 games, about
   +5 points of pooled win rate at the current pairing level.
3. **Stop optimising tempo and attack-count metrics.** They are measured nulls
   at n = 194.
4. **Ladder protocol.** Submit challenger and control into the two slots
   simultaneously and submit nothing else until both finish. With a 77 Elo
   floor and a 74 Elo target, sequential single runs cannot resolve the
   question.
