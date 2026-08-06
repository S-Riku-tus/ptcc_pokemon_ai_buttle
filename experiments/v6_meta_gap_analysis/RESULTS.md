# Where v6 stands against the current top of this archetype (2026-08-06)

v1-v6 were all ranked by one pooled ladder rating and fixed by comparing
per-decision imitation rates against the corpus pilots. Both instruments are now
exhausted: no measured behaviour rate orders the pilots by rating, and v6's own
remaining divergences are inside the top pilots' band. This report changes the
unit of analysis from *decisions* to *matchups*, because that is where the
remaining rating is, and it reports three hypotheses that measurement killed.

All numbers come from data already in the repo: 4,186 same-deck corpus games
(deck hash `9714ab5c3996f6cc`, self-play validation episodes excluded), our own
ladder runs for v1-v5, and the 2026-08-06 top-60 leaderboard snapshot.

Scripts and outputs live beside this file:

| script | output | what it measures |
| --- | --- | --- |
| `measure_matchup_gap.py` | `matchup_gap.json` | win rate per opponent deck hash, per label |
| `name_archetypes.py` | `archetypes.json` | deck hash to headline Pokemon, top-40 weight, our record by opponent rating |
| `measure_matchup_shape.py` | `matchup_shape_corpus.json` | wins vs losses inside each matchup |
| `measure_setup_pace.py` | `setup_pace.json` | turn Grimmsnarl ex reaches play, per pilot and for us |
| `measure_meta_pressure.py` | `meta_pressure.json` | slot-weighted expectation and deficit attribution |
| `measure_ogerpon_exposure.py` | `ogerpon_exposure.json` | whether our own Energy makes the counter lethal |

## 1. Our ladder rating is a fixed point, not a measurement of the policy

Pooled win rate says our recent versions beat the top pilots: v4 0.627, v4.5
0.702, v5 0.661, against 0.537 for rank-5 16452116 and 0.574 for the whole
corpus field. That comparison is worthless, and the reason is in the episode
index.

| our run | games | win rate | mean opponent rating |
| --- | --- | --- | --- |
| v4 `sub55253296` | 51 | 0.627 | 967.1 |
| v4.5 `sub55275464` | 67 | 0.702 | 843.0 |
| v5 `sub55275642` | 65 | 0.661 | 859.6 |

Pooled over v4/v4.5/v5 (183 games), banded by the opponent's pre-game rating:

| opponent rating | games | win rate |
| --- | --- | --- |
| < 900 | 74 | 0.757 |
| 900-1000 | 85 | 0.635 |
| 1000-1050 | 20 | 0.450 |
| 1050-1100 | 4 | 0.750 |

We are matched against a ~860-rated pool, win three quarters of it, and fall to
0.450 as soon as the opponent is over 1000. A ladder run therefore cannot tell us
anything about play at 1100+, and 65 games per run cannot separate two versions
whose stored-board agreement is 97.8%.

## 2. The field turned over, and we have never played most of the current top 20

Of the current top 40 submissions, 6 play this exact 60-card list (best rank 5,
1142.7), against 51% of the top 50 four days earlier. The rest of the top 40 is
what we now have to beat, and our ladder pool has almost no overlap with it.

| opponent deck | top-40 slots | best rank | field games | field win | top-3 pilots | our games |
| --- | --- | --- | --- | --- | --- | --- |
| `cc38cb450b86770a` Alakazam + Kadabra | 6 | 7 | 268 | **0.724** | 0.68 | 23 |
| `9714ab5c3996f6cc` mirror | 6 | 5 | 1387 | 0.578 | 0.55 | 55 |
| `a7ee29914c1dce64` Mega Lopunny ex + Mega Froslass ex | 5 | 3 | 106 | **0.415** | 0.42 | **0** |
| `711c221c9cc49384` Thwackey + Dipplin | 2 | 23 | 100 | 0.480 | 0.47 | 1 |
| `6fa64c0e3b2eb67c` Mega Lopunny ex + Dudunsparce | 2 | 17 | 213 | 0.512 | 0.43 | 1 |
| `202ee2cec6cbe8b4` Dragapult ex | 2 | **1** | 45 | 0.511 | 0.43 | **0** |
| `05f854e1b3b3ba0f` Mega Lucario ex | 2 | 11 | 49 | 0.592 | 0.59 | 1 |
| `20bc24847121c967` Mega Kangaskhan ex | 2 | 20 | 19 | 0.684 | - | 3 |
| `0dede7cb8026e473` Teal Mask Ogerpon ex | 1 | 33 | 218 | **0.183** | 0.20 | 5 |
| `97df7a2a423da1d8` Teal Mask Ogerpon ex (Tera Orb) | 1 | 39 | 52 | **0.231** | 0.22 | 1 |
| `3f1fae2704f11912` Mega Lopunny ex + Lillie's Clefairy ex | 1 | 37 | 73 | 0.603 | 0.55 | 1 |
| `1eda048ea3581634` Mega Kangaskhan ex | 1 | 35 | 33 | 0.697 | 1.00 | 4 |

Slot-weighting those cells (cells with at least 20 field games; our own
submission removed from the mirror's six slots) gives the number that matters for
the next ladder run:

* **expected win rate against the current top-40 field: 0.543**
* using the top-3 same-deck pilots' own rates instead: **0.515**

The whole surplus is the Alakazam matchup: 21.4% of the slot-weighted field at
0.724. Three cells carry the entire deficit.

| deck | share | field win | deficit vs even |
| --- | --- | --- | --- |
| `a7ee29914c1dce64` | 0.179 | 0.415 | **-0.0152** |
| `0dede7cb8026e473` | 0.036 | 0.183 | -0.0113 |
| `97df7a2a423da1d8` | 0.036 | 0.231 | -0.0096 |

So there are about 3.6 points of win rate on the table, and they are in two
archetype families, not in a MAIN preference.

## 3. Three hypotheses that measurement killed

Each of these looked mechanically compelling from the card text. Two are wrong
and the third is real but not worth acting on; all three are recorded so nobody
spends a version on them.

**Hand size against Mega Froslass ex - refuted.** Resentful Refrain costs one
{W} and does 50 damage per card in our hand, which would make a 7-card hand
lethal on a 320 HP Grimmsnarl ex. In the 106 corpus games against
`a7ee29914c1dce64` the winners hold the *bigger* hand: mean 4.476 vs 3.836, and
hands of 7+ on 12.5% of turns vs 5.3%. The hand size on the turn before we lose a
Pokemon is also higher in wins (4.897 vs 3.870). A large hand is a symptom of a
working engine, and the mean hand of 4.5 only pays for 225 damage.

**Punk Up feeding Teal Mask Ogerpon ex - refuted as a lever.** Myriad Leaf Shower
is 30 damage plus 30 for each Energy attached to *both* Actives, and Marnie's
Grimmsnarl ex, Morgrem and Impidimp all have Grass weakness, so the doubled total
includes the Dark Energy we attached ourselves - and Punk Up attaches up to five
while Shadow Bullet only costs two. Reconstructing every turn we hand to an
Ogerpon deck (271 games):

| | all | wins | losses |
| --- | --- | --- | --- |
| turns facing an Active Ogerpon | 4.72 | 4.29 | 4.82 |
| Energy on our own Active (mean) | 1.28 | 1.61 | 1.21 |
| their Active can OHKO ours | 0.802 | 0.741 | 0.816 |
| ... if our Active held at most 2 | 0.792 | 0.713 | 0.811 |
| ... with one more Teal Dance Energy | **0.939** | 0.906 | 0.947 |
| ... capped *and* with Teal Dance | 0.935 | 0.895 | 0.944 |

We already only hold 1.28 Energy on the Active, so capping it is worth 0.4
points. The real finding is the row above it: **on 93.9% of the turns we pass to
an Ogerpon deck, their Active can already kill ours.** Grass weakness means three
Energy is enough. This matchup is a deck-level loss, not a play error, which is
why the field, the elite pilots and we all sit at 0.18-0.23.

**Setup pace as a global target - real but already matched.** The turn Grimmsnarl
ex first reaches play is the one metric that separates wins from losses in *every*
current matchup: mirror 4.83 vs 5.12, Alakazam 4.77 vs 5.78, Ogerpon 4.20 vs
5.57, Lopunny/Dudunsparce 4.28 vs 5.34, Lopunny/MFroslass 4.32 vs 5.10, Dragapult
5.96 vs 7.60, and `grimmsnarl_ex_ever` is ~1.00 in wins against 0.92-0.96 in
losses. But measured per pilot it has no rating gradient (16371703 at 1220.2 is
on 4.585, the field mean is 4.543, our pin 16494330 at 1077.6 is on 4.348), and
our own pace is already at or ahead of the field: v5 reaches it on own turn 4.587
with 38.5% by own turn 3, against the field's 4.543 and 31.0%. The within-game
correlation is at least partly reverse causality - a bad opening produces both a
late evolve and a loss.

Two smaller checks, for the record: v5's going-second gap has closed (0.676
first / 0.645 second over 65 games, against the field's 0.610 / 0.536), and
against Ogerpon the wins are the short games (ending turn 7.70 vs 9.19) with
*fewer* Boss's Orders (0.08 vs 0.37), i.e. racing rather than disrupting.

## 4. What is actually available to fix

**4.1 Build a local opponent panel; it is the only instrument that can rank a
change.** Our replays contain the opponent's own observations and actions, not
just ours - 76 to 109 selects per game on their side. So the same imitation
pipeline that produced the Grimmsnarl ranker can produce sparring partners from
data we already hold: `0dede7cb8026e473` (218 games), `6fa64c0e3b2eb67c` (213),
`a7ee29914c1dce64` (106), `202ee2cec6cbe8b4` (45). Local arena games are
unlimited and paired, which is the opposite of a 65-game ladder run against an
860-rated pool. Everything below depends on this existing first.

**4.2 Target `a7ee29914c1dce64` - the biggest recoverable cell.** 5 of the
current top 20, 17.9% of the slot-weighted field, 0.415 for the field and 0.417
for the top-3 pilots, and we have never played it. Because the entire field is at
42%, imitation cannot supply the answer: this is planner and evaluation work, and
the 106 stored games are a ready-made counterfactual probe target. Their threats
are Mega Lopunny ex (Gale Thrust, 230 for one Energy if it moved from the bench
this turn, so it snipes the 70-110 HP support line, plus Spiky Hopper 160 that
ignores effects on our Active) and Mega Froslass ex (Absolute Snow 150 and
Asleep). Note the direction of the extra evidence: on 98 games the top-3 pilots
are at 0.43 against the closely related `6fa64c0e3b2eb67c`, below the field's
0.512, so this family may be worse than the headline cell.

**4.3 Treat the Ogerpon family as unfixable by policy and price it in.** 3 of the
top 40, ~7% of the slot-weighted field, 2.1 points of expected win rate, and
93.9% of our passed turns are already lethal. The only lever the data supports is
finishing the game before turn 8, which is the same lever as 4.2's setup pace and
is not specific to this matchup.

**4.4 Refresh the corpus, for the field's composition rather than for accuracy.**
559 new same-deck games are available (about +13.6%), the corpus reflects the July
field, and all three current top same-deck pilots are in it but capped at 300
games while 885, 791 and 303 episodes are now available. This was already the
documented trigger; the new argument for it is that the July field it encodes is
not the field a v7 would play.

**4.5 The strategic fork, which is a decision rather than a task.** This
archetype has fallen from 51% of the top 50 to 6 of the top 40 in four days, its
expectation against the current top-40 field is 0.543 - and that rests on a
0.724 Alakazam cell whose own share is shrinking. Ranks 1 through 4 are Dragapult
ex and Mega Lopunny ex + Mega Froslass ex. Starting an imitation line on
`a7ee29914c1dce64` would reuse the whole pipeline unchanged and has 5 top-20
pilots and 646/217/154 episodes available to learn from. Continuing on Grimmsnarl
means accepting a coin-flip archetype with two unwinnable cells.

## 5. What not to do

* Another MAIN preference pin. No measured behaviour rate orders the 21 pilots
  by rating, v6 is already inside the top pilots' band on the Froslass evolve,
  and three previous versions moved nothing with feature columns.
* Hand-size management against Mega Froslass ex (section 3, refuted).
* Spreading Punk Up Energy to dodge Myriad Leaf Shower (section 3, 0.4 points).
* A global setup-pace push: we are already at or ahead of the field.
* A going-second fix: v5's split is 0.676 / 0.645 over 65 games.

## 6. Limitations

* Field win rates come from the July corpus. Those archetypes' current lists may
  differ from the representative list used for naming, and the pilots behind them
  have changed submissions.
* Slot weighting treats one top-40 submission as one unit of field share, which
  ignores how many games each actually plays.
* The banded win rates in section 1 have 20 games in the 1000-1050 bucket and 4
  above it.
* Wins-versus-losses splits inside a matchup are correlational. Section 3 treats
  the setup-pace split accordingly and the Ogerpon exposure result does not
  depend on any such split.
