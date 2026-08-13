# v19 / v20 ladder analysis (2026-08-12)

Artifacts: `ladder_v19_v20.json`, `ladder_v19_v20_games.csv`,
`ladder_history.json`, `ladder_history_games.csv`, `field_turn_order.json`,
`rating_levers_v19.json`, `rating_levers_v20.json`, `slow_games.json`.
Scripts: `scripts/analyze_grimmsnarl_v20_ladder.py`,
`scripts/analyze_grimmsnarl_field_turn_order.py`,
`scripts/analyze_grimmsnarl_v20_slow_games.py`.

## 1. v20 beats v19; the raw win rate is a pairing artifact

| | games | record | win rate | mean opponent rating |
| --- | ---: | ---: | ---: | ---: |
| v19 | 43 | 27-16 | 0.628 | 842.2 |
| v20 | 47 | 28-19 | 0.596 | 910.8 |

v20 met a much stronger pool. Bucketed by the opponent's rating at pairing:

| opponent band | v19 | v20 | Fisher p |
| --- | ---: | ---: | ---: |
| <900 | 23-8 | 12-3 | 1.00 |
| 900-1000 | 3-8 | 15-6 | **0.027** |
| >=1000 | 0-0 | 1-9 | - |

v19 never faced a 1000+ opponent at all. On the only band where both have a
denominator, v20 is better with p = 0.027. The 77-point rating difference is
not itself evidence (see `kaggle-ladder-rating-noise`); this comparison is.

## 2. There is no collapse against strong opponents

The naive read of v20's 1-9 versus 1000+ looks like a ceiling, and bucketing
by *our own* rating at pairing appears to confirm it (Elo residual +0.330 below
700, +0.024 at 900-1000, **-0.133** at >=1000, n=87). That conditioning is
mean-reversion: a rating is high exactly when recent luck was good.

Re-estimating each version's strength from its **final** rating instead removes
the self-conditioning, and the residuals go flat:

| opponent band | n | win rate | Elo expectation | delta |
| --- | ---: | ---: | ---: | ---: |
| <800 | 118 | 0.805 | 0.787 | +0.018 |
| 800-900 | 149 | 0.644 | 0.575 | +0.069 |
| 900-1000 | 187 | 0.519 | 0.508 | +0.011 |
| >=1000 | 64 | 0.422 | 0.415 | +0.007 |
| all | 518 | 0.608 | 0.579 | +0.029 |

Across v13-v20 we are 27-37 (0.422) versus opponents rated 1000+, which is
exactly what a ~950-rated agent should score. v20's 1-9 is a 10-game sample on
top of a rating that had peaked at 1029.3 mid-run.

## 3. The going-second gap is closed

| cohort | first | second | gap |
| --- | ---: | ---: | ---: |
| ours, v15-v20 (368 games) | 0.653 | 0.575 | +0.078 |
| field, all pilots (3,642 games) | 0.6209 | 0.5556 | +0.065 |
| field, 1100+ pilots (1,323 games) | 0.6095 | 0.5453 | +0.064 |

Our split is the deck's own structural turn-order penalty, not a defect.
Per-pilot field gaps run -0.092 to +0.193, so +0.078 is unremarkable.
The old top priority - Alakazam going second - is finished: v15-v20 is
24-20 there with an Elo residual of -0.001, and 29-6 (+0.258) going first.

## 4. Ogerpon is the only archetype with a real deficit

Elo residual by opponent family, v13-v20 (n>=8):

Ogerpon is the known structural counter (Grass weakness on the whole Marnie
line) and is not policy-fixable. Everything else is at or above expectation.

## 5. The one live gradient: when Grimmsnarl ex reaches the board

| own turn Grimmsnarl ex first on board | n | win rate |
| ---: | ---: | ---: |
| 2 | 209 | 0.689 |
| 3 | 222 | 0.613 |
| 4 | 58 | 0.534 |
| 5 | 22 | 0.364 |
| never | 18 | **0.000** |

Board and attack-ready are the **same own turn in 511 of 511 games** - Punk Up
funds the attacker at the moment it evolves - so there is no energy bottleneck
and the whole race collapses to one variable. 20 games never used Shadow
Bullet at all and won none of them.

53 games (10.0%) are slow: first Shadow Bullet on own turn 5 or later, or
never. Of the 35 slow games that did assemble a Grimmsnarl, the 80 own turns
between "ready" and "first Shadow Bullet" decompose as:

| cause | turns |
| --- | ---: |
| ready Grimmsnarl **on the bench**, Active cannot retreat | 32 |
| Grimmsnarl Active, Shadow Bullet legal, **not taken** | 19 |
| Grimmsnarl Active, Shadow Bullet not legal (Crustle wall) | 1 |
| on the bench, retreat available and unused | 1 |

Every one of the 32 bench-locked turns had a **0-energy** Snorunt (19),
Munkidori (10) or Froslass (3) as the Active. All three retreat for 1 and none
is a Marnie's Pokemon, so Punk Up cannot fund them; in 31 of the 32 turns an
energy attachment was no longer a legal option that turn. Corpus-wide the
state occurs on **128 own turns across 82 of 529 games (15.5%)**.

## 6. Ruled out - do not retry

* **Mis-attaching the manual Dark Energy.** Of 1,881 manual attachments, only
  **4** (in 3 games) went to a Marnie body while the Active was a 0-energy
  non-Marnie body with a Marnie line benched. The bench-lock is not caused by
  a wrong attach target at that moment; the energy simply is not there.
* **Promotion after a KO.** When a ready Grimmsnarl ex was an option:
  252/252 = 100% in the forced-promotion context, 397/420 = 94.5% in the
  switch context, and 18 of the 23 exceptions promote Munkidori.
* **Evolve target.** 362 of 363 bench evolutions had no legal Active target;
  exactly one chose the bench while an Active target was legal. 298 of them
  had a Grimmsnarl ex already Active (building the second attacker).
* **Opening Active choice.** Impidimp 0.603 (n=295), Munkidori 0.644 (135),
  Snorunt 0.545 (99). No gradient, and 290 of 529 openings were forced.
* **Spikemuth Gym search targeting.** 3 redundant Grimmsnarl ex searches in
  529 games; targets split grimmsnarl 816 / morgrem 610 / impidimp 494.

## 7. Style axis versus the field rating gradient

From `rating_levers_*.json` (21 archived pilots, BH-controlled):

| statistic | dir | elite | rest | v19 | v20 |
| --- | --- | ---: | ---: | ---: | ---: |
| attack_taken | lower | 0.9407 | 0.9574 | 0.9532 | **0.9670** |
| boss_taken | higher | 0.3847 | 0.3631 | 0.3026 | **0.1795** |
| supporter_is_boss | higher | 0.1601 | 0.1498 | 0.1565 | 0.0909 |
| supporter_taken | lower | 0.8580 | 0.8794 | 0.8258 | 0.7817 |
| petrel_boss | higher | 0.0931 | 0.0458 | 0.0612 | 0.0566 |
| bench_full_at_turn_end | lower | 0.8744 | 0.8912 | 0.9027 | 0.8917 |
| retreat_taken | higher | 0.2127 | 0.1712 | 0.2216 | 0.2217 |
| win_rate | - | 0.5903 | 0.6081 | 0.6190 | 0.6087 |

`retreat_taken` is fixed. `attack_taken` moved back toward v8's commit-heavy
0.9683 and `boss_taken` halved - both away from the elite band. Our raw win
rate already exceeds the elite band's, which restates the standing result that
rating on this ladder does not track win rate.

## 8. Recommendations

1. **Promote v20 over v19** on the matched-band comparison, not on rating.
2. **Attack the slow tail, nothing else.** Two sub-targets with a measured
   footprint: the 128 bench-locked turns in 82 games, and the 19 refused
   Shadow Bullets. Both are inside the 10% of games that are worth roughly
   0.35 win rate each.
3. For the bench-lock, the unlock has to be **one turn earlier** than the lock
   - reserve the turn's Dark Energy for a non-Marnie Active whenever a Marnie
   line is benched and the Active holds none. Sweep every legal input for the
   footprint before spending a ladder slot; both v18 guards bound 0 times.
4. For the 19 refusals, add that state to the hard-state weighting rather than
   writing a rule.
5. **Do not touch** Ogerpon, the mirror, turn order, the opening Active,
   promotion, evolve targets or gym targeting - all measured flat here.
6. If v21 retrains, **control `attack_taken` and `boss_taken`** against v19 at
   equal patience; v20 regressed on both.
