# v24 ladder verdict — 87 games over two submissions, 2026-08-14

Submissions `55496021` (911.3, 39 games) and `55496665` (928.1, 48 games),
pooled with v22's four byte-identical runs (194 games) for a 281-game corpus.

Scripts: `analyse_v24_ladder.py`, `analyse_v24_guard_binding.py`,
`analyse_rating_trajectory.py`, `analyse_elo_income.py`,
`analyse_strong_band.py`, `analyse_ogerpon_and_mirror.py`,
`analyse_mirror_race.py`, `analyse_compute_gap.py`, `analyse_grass_exposure.py`,
`analyse_policy_elo.py`, `probe_top60_decks.py`.

---

## 1. The experiment bound perfectly and bought nothing

V24's only change was a veto of Froslass evolution in the visible mirror.
On the ladder it did exactly what the offline footprint promised:

| | v22 (194 games) | v24 (87 games) |
|---|---:|---:|
| mirror games | 56 | 28 |
| true Froslass evolutions in the mirror | 0.34 / game | **0.000 / game** |
| mirror games with a surviving evolution | — | **0 of 28** |

100% binding, zero leakage, no non-mirror change. And no effect:

| instrument | result |
|---|---|
| `is_v24` controlled for opponent rating and turn order | **−72.7 Elo, p = 0.178** |
| v24 finals vs v22 finals | 911.3 / 928.1 vs 1000.6 / 1018.6 / 1020.4 / 952.8 |
| pooled strength (see §4) | v22 1008.8 [958, 1060], v24 930.7 [853, 1008] |

Two independent instruments agree on −73 to −78 Elo. The intervals overlap, so
this is "no better, plausibly worse", not a proven regression — but the
direction is the wrong one and v22 remains champion.

## 2. Why it bought nothing: the lever was a confound

The contrast that motivated v24 (mirror games with a Froslass evolution win
0.294 vs 0.641 without) survives *inside v24*, in the games the guard cannot
touch:

| pool | true-evolution contrast, controlled |
|---|---:|
| v22, all games | −41.1 Elo, p = 0.47 |
| **v24, all games (mirror events = 0)** | **−191.7 Elo, p = 0.0415** |

Removing every mirror evolution did not remove the association; it reappeared
in the non-mirror games, where v22's own placebo check had previously found
nothing. A cause that is deleted and whose signature stays is not a cause.
`froslass_evolves > 0` is a marker of a game that went long on a bad board.

**This retires the whole method, not just this lever.** Twenty-plus behavioural
levers have now been read off win-rate splits on this corpus; v24 is the first
one that was actually intervened on, and it falsified the reading.

## 3. 928 is a converged number, not a truncated one

Per-submission trajectories (`rating_trajectory.json`), all ten Grimmsnarl runs:

| run | n | win rate | opp mean | final | implied equilibrium | gap |
|---|---:|---:|---:|---:|---:|---:|
| v22_a | 45 | 0.733 | 836 | 1000.6 | 1012.2 | 11.6 |
| v22_b | 38 | 0.711 | 885 | 1018.6 | 1040.8 | 22.2 |
| v22_c | 57 | 0.544 | 1000 | 1020.4 | 1030.7 | 10.3 |
| v22_d | 50 | 0.580 | 912 | 952.8 | 968.5 | 15.7 |
| v24_a | 38 | 0.684 | 797 | 911.3 | 931.4 | 20.1 |
| v24_b | 47 | 0.638 | 832 | 928.1 | 930.8 | 2.7 |

Every run lands within 3–22 Elo of `opponent mean + 400·log10(w/(1−w))`. More
games would not have moved either v24 number. Equally, the raw win rate is not
the story: v24 looks *better* than v22 (0.655 vs 0.624) purely because the
pairing draw handed it a 100-point weaker field (816 vs 915).

## 4. Where the rating is actually earned

Exact decomposition of the per-episode `updatedScore − initialScore`:

| opponent band | n | record | win rate | net Elo | Elo/game |
|---|---:|---:|---:|---:|---:|
| < 700 | 35 | 31-4 | 0.886 | **+978** | +27.9 |
| 700–800 | 24 | 17-7 | 0.708 | +364 | +15.2 |
| 800–900 | 58 | 45-13 | 0.776 | +497 | +8.6 |
| 900–1000 | 101 | 54-47 | 0.535 | +311 | +3.1 |
| 1000–1100 | 50 | 27-23 | 0.540 | +127 | +2.5 |
| ≥ 1100 | 7 | 2-5 | 0.286 | −46 | −6.5 |

The climb from Kaggle's 600 start is financed almost entirely by sub-900
opponents. Those pairings dry up as the rating rises, and from 900 up the
policy earns ~2.5–3.1 Elo per game at a 53–54% win rate. **The equilibrium is
the ≥950 win rate and nothing else.**

Pooled strength estimates (`analyse_policy_elo.py`):

| pool | n | win rate | opp mean | implied strength |
|---|---:|---:|---:|---:|
| all v22+v24 | 275 | 0.640 | 884.7 | **984.6 [942, 1027]** |
| opponents ≥ 950 | 109 | 0.523 | 1012.9 | 1028.8 [964, 1094] |

Same-deck peers on the current board — the probe finished 49 of 60 ranks, and
`9714ab5c3996f6cc` holds **five** of them:

| rank | rating | team | submission |
|---:|---:|---|---|
| 22 | 1095.3 | AlphaTCG (16381823) | 55350342 |
| 27 | 1086.0 | NguyenThanhNhan (16381904) | 55501065 |
| 44 | 1054.7 | やる気元気ミワハルキ (16371703) — **our pinned teacher** | 55494672 |
| 54 | 1040.1 | Phil_Hellmuth (16383701) | 55454605 |
| 57 | 1037.9 | Roman Nesterov (16507505) | 55500801 |

Five independent pilots cluster in **1037–1095**, which is a much tighter
ceiling estimate for this 60 than any single number. Our pooled 984.6 sits
below all five. To sit where the best of them sits we need **0.616 against
≥950 opponents, i.e. +10 wins in 109 games**.

## 5. Where those 10 games are — and are not

Opponents ≥ 950, 109 games, 0.523:

| matchup | n | record | win rate |
|---|---:|---:|---:|
| Grimmsnarl (mirror) | 31 | 14-17 | **0.452** |
| Mega Lopunny / Froslass | 18 | 9-9 | 0.500 |
| Dragapult | 17 | 10-7 | 0.588 |
| Kangaskhan / Crustle | 12 | 8-4 | 0.667 |
| Alakazam | 12 | 9-3 | 0.750 |
| **Ogerpon** | 10 | **0-10** | **0.000** |

**Ogerpon is confirmed unfixable and should be priced in, not worked on.** All
13 Ogerpon games (1-12 overall) are against the pure Teal Mask shell: 4–5
Pokemon, every one of them Grass, against three bodies that all carry
`weakness == 1` (Grass, ×2). `turns_with_immune_active = 0` in every game, so
this is not the Cornerstone wall that `wall_break` addresses — it is the
straight weakness race, at the ~0.20 the whole field scores. It is also *not* a
top-40 deck, so it taxes the climb but does not block 1100.

**Tempo is dead, definitively.** In the mirror we start attacking **2+ turns
before the opponent in 60% of games** (18 of 31 at ≥950) and still win 0.500 in
exactly those games. `own_first_shadow_turn` is 2.96 in wins and 2.96 in
losses. Every controlled term inside the mirror is null: attack lead +52 Elo/turn
p=0.16, Grimmsnarl evolutions +165 p=0.16, Stamp −35 p=0.65, Boss +79 p=0.29.
We get there first and lose the exchange that follows.

## 6. The blind spot: 19% of the top 40 is unmeasured

Top-40 deck hashes decoded from live replays (`top40_decks_20260814.csv`):

| deck | slots | ranks | best | our record |
|---|---:|---|---:|---:|
| `202ee2cec6cbe8b4` Dragapult ex | 9 | 1, 11, 13, 14, 18, 24, 32, 38, 40 | 1254 | 9-5 (0.643) |
| `f39b7a1dd837f526` Ogerpon + Hydrapple ex (17/20 Grass) | 5 | **4, 6, 8**, 26, 36 | 1196 | **0-2** |
| `a7ee29914c1dce64` Mega Lopunny / Froslass | 4 | 23, 29, 31, 35 | 1093 | 11-7 (0.611) |
| `9714ab5c3996f6cc` **ours** | 2 | 22, 27 | 1095 | — |
| `82879abd3807469b` Ogerpon + Arboliva (17/20 Grass) | 1 | **3** | 1204 | 0-1 |
| `63e16702d79a67b5` Ogerpon + Hydrapple | 1 | **10** | 1158 | 0-0 |

The single biggest top-40 archetype (Dragapult, 22% of slots including rank 1)
is a matchup we **win** at 0.643. The Grass midrange shell holding ranks
3/4/6/8/10 is one we have **2–4 games against, all told, and are 0-3 in**. That
is the largest unmeasured slice of the road to 1100, and it shares its damage
mechanism with the Ogerpon cell we are proven to lose.

## 7. Unused compute

Episode configuration is `actTimeout: 0` with a **600-second** per-episode
overage bank.

| | mean | max | share of bank |
|---|---:|---:|---:|
| us | 12.0 s | 21.8 s | **2.0%** |
| opponents | 33.4 s | 439.2 s | 5.6% |
| opponents rated 1000–1100 | 95.5 s | 543.9 s | 30% spend ≥60 s |

v22/v24 are a one-ply ranker plus rule fallback with no lookahead, no deadline
and no timing code at all. **Caveat, stated plainly: opponent compute spend
does *not* predict our losses once rating is controlled** (0.0 Elo, p = 0.85;
the raw buckets are non-monotone). This is an unused resource, not evidence
that search wins. It is the only large untried input we have.

## 8. Peer check — and a correction to §6

120 replays of **AlphaTCG** (team 16381823, submission 55350342, rank 22,
**1095.3**, identical 60) were pulled after the rate limit cleared
(`analyse_peer_gap.py`). Their corpus is 2026-08-10..14, ours 08-13..14, so
compare cells and not pooled numbers.

Their pooled win rate is **0.542**, *lower* than our 0.633 — because 109 of
their 120 games are against opponents rated ≥1000, versus 57 of our 281.

| opponent band | us | AlphaTCG |
|---|---:|---:|
| < 900 | 0.795 (n=117) | — (n=4) |
| 900–1000 | 0.535 (n=101) | 0.800 (n=5) |
| **1000–1100** | **0.540 (n=50)** | **0.571 (n=49)** |
| ≥ 1100 | 0.286 (n=7) | **0.467 (n=60)** |

**In the one band where both corpora have a real sample we are
indistinguishable from the 1095-rated pilot** (0.540 vs 0.571, Wilson intervals
almost coincident). Below 900 we score 0.795 where a 985-strength player is
*expected* to score 0.797 — exactly on model, so there is no cheap leak there
either. Every behaviour count at ≥950 is within ±0.6 of theirs (attacks 4.08 vs
3.72, Adrena-Brain 6.12 vs 5.49, first Shadow turn 2.96 vs 3.16), and their
win/loss splits show the same scoreboard pattern ours do.

So **"we play ~110 Elo worse than the same deck's best pilot" is not
established.** What is established is where their rating lives: 60 games
against ≥1100 opponents held at 0.467, a band we have seen 7 times.

Matchup cells at ≥950 where the two corpora genuinely differ:

| matchup | us | AlphaTCG |
|---|---:|---:|
| Grimmsnarl (mirror) | 0.452 (n=31) | **0.750 (n=8)** |
| Mega Lopunny / Froslass | 0.500 (n=18) | 0.643 (n=14) |
| Dragapult | 0.588 (n=17) | 0.513 (n=39) |
| **Ogerpon + Hydrapple (Grass midrange)** | 1.000 (n=2) | **0.250 (n=12)** |
| Mega Lucario | — (n=0) | 0.222 (n=9) |

**This corrects §6.** The Grass midrange holding ranks 3/4/6/8/10 is *not* a
fixable blind spot: the 1095-rated pilot of our own deck is **3-9** against it.
Like Ogerpon, it is a tax to price in, not a cell to work on. Mega Lucario at
≥950 (0.222 for them, unmeasured for us) is the remaining genuine unknown.

The mirror stays the one cell where a better same-deck pilot visibly separates
from us — consistent with §5, where we showed it is not a tempo problem.

---

## Recommendations, in priority order

1. **Keep v22 as champion; do not promote v24.** Retire the Froslass lever.
2. **Stop reading levers off win-rate splits on this corpus.** v24 is the
   falsification: the one lever that was actually intervened on was a confound.
   Any future behavioural claim needs an intervention, not a contrast.
3. **The mirror at ≥950 (31 games, 0.452 vs the peer's 0.750) is the one cell
   where a better same-deck pilot visibly separates from us, and it is not a
   tempo problem.** We attack first and lose the exchange. That is a multi-turn
   evaluation question a one-ply scorer cannot express — search or outcome
   learning, with 98% of a 600-second bank sitting idle.
4. **Price in Ogerpon *and* the Grass midrange; do not work on either.** The
   1095-rated pilot of our own list is 3-9 against the Grass midrange and 0-2
   against Ogerpon. Together they are ~25% of the top 40 at ~0.20-0.25 for
   anyone playing this deck.
5. **Put the 60 cards on the table for the first time.** Every version since v1
   has frozen `9714ab5c3996f6cc`. The best pilot of that list is at 1095 while
   rank 1 is at 1254 on Dragapult. Perfect play on this deck is worth ~1100.
