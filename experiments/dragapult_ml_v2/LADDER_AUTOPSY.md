# Why the imitation agent is not rated like the pilots it imitates

Submissions 55545828 (v1.0) and 55550682 (v2), audited on 2026-08-16 against
the 1,392-game exact-list teacher corpus.

**Revision 2** — v2's log count grew from 12 to 26 games. Three conclusions
from revision 1 did not survive the larger sample and are corrected below:
the loss shape, the size of the OOD defect, and the value of relaxing the OOD
gate. The Phantom Dive law survived, out of sample.

## 0. What the ladder actually says

| | v1.0 | v2 |
|---|---:|---:|
| public games | 22 | 25 |
| record | 11-11 | 14-11 |
| opponent mean initial rating | 493.8 | 661.1 |
| displayed rating | 507.8 | 701.2 |
| Elo fixed point (opp mean + Elo(wr)) | 493.8 | 703.0 |

v2 is **+209 Elo** on v1 at the fixed point, and both ran on 2026-08-16, so
calendar drift is not in the comparison. v2's per-game rating step has decayed
from ±137 to ±13, and its second half is 8-5 against a *higher* mean opponent
(674.5) than its first half (646.6) — it is still converging upward, not stuck.
The "about 650" the run was judged on was an 11-game reading.

Opponent-bucketed: 500-600 4-0, 600-700 7-8, 700-800 3-3.

## 1. The win rate is a function of one variable — confirmed out of sample

Over 1,392 teacher games, win rate is determined by how many Phantom Dives the
game yields. The v2 run reproduces the same conditionals on data that was not
available when they were fitted:

| Phantom Dives | teacher n | teacher wr | v2 n | v2 wr |
|---|---:|---:|---:|---:|
| 0 | 83 | 0.108 | 4 | 0.000 |
| 1 | 121 | 0.231 | 4 | 0.250 |
| 2+ | 1,188 | 0.731 | 18 | 0.722 |
| (4+) | 505 | 0.867 | 9 | 0.778 |

Applying the teacher conditionals to each agent's own histogram:

| | P(0) | P(1) | P(2+) | P(4+) | mean PD | implied wr | actual wr |
|---|---:|---:|---:|---:|---:|---:|---:|
| teachers | 0.060 | 0.087 | 0.853 | 0.363 | 3.084 | 0.650 | 0.651 |
| v2 (26 games) | 0.154 | 0.154 | 0.692 | 0.346 | 3.038 | 0.558 | 0.538 |
| v1.0 (23 games) | 0.391 | 0.130 | 0.478 | 0.130 | 2.043 | 0.422 | 0.522 |

**The mean and the upper tail are already at teacher level** — 3.038 dives per
game against 3.084, and P(4+) 0.346 against 0.363. The entire deficit is the
low tail: P(0)+P(1) is **0.308 against the teachers' 0.147**. Closing only that
is worth 0.650 − 0.558 = 0.092 win rate, about **+67 Elo**.

Revision 1 reported v2's losses as 100% blow-outs at 0.167 prizes taken. On 26
games they are 66.7% blow-outs at 1.333 prizes, against the teachers' 60.9% and
1.959. The loss *shape* is close to normal; the loss *count* is not.

## 2. The tail is not a draw problem and not an arrival problem

Eight of 26 games yielded <=1 dive. Only one was a dead draw (93609360, three
own turns). Six had Dragapult ex on the board and attacked with something else.

| | tail (8) | rest (18) |
|---|---:|---:|
| own turns | 7.9 | 11.2 |
| first Dragapult ex on own turn | 6.6 | 5.6 |
| own turns with Dragapult ex Active | **1.6** | **4.9** |
| own turns with it powered (Fire+Psychic) | **0.6** | **4.6** |
| search take rate | 0.292 | 0.185 |
| prizes taken / conceded | 0.6 / 3.0 | 3.9 / 2.7 |

Dragapult ex *arrives* one turn later in the tail. It is *usable* for a quarter
as long. And the search rate is higher in the tail, so it is not that the agent
forgot to dig.

## 3. Energy readiness is largely fixed; board width is not

Per own turn with Dragapult ex Active:

| | turns/game | dive-ready | ready but did not dive |
|---|---:|---:|---:|
| teachers | 3.81 | 0.925 | 0.089 |
| v2 | 3.92 | 0.853 | 0.092 |
| v1.0 | 4.39 | 0.475 | 0.083 |

v2 closed most of v1's typed-energy hole (0.475 -> 0.853) and already puts
Dragapult ex in the Active slot as often as the teachers. What it does not do
is *build* as many:

| per game | teachers | v2 |
|---|---:|---:|
| Dragapult ex created | **1.883** | **1.577** |
| created on the bench | 1.192 | 0.769 |
| created in the Active slot | 0.690 | 0.808 |
| of those created ever using Phantom Dive | 0.841 | 0.732 |

## 4. The one clean policy defect: evolving into an Active it cannot power

Trace of episode 93603391 (a Mega Lucario loss, nine own turns, zero attacks):
on own turn 5 the agent evolved its Active Drakloak into Dragapult ex with no
Energy in hand and no attach available. It was knocked out immediately for two
prizes. A Drakloak would have cost one.

| share of Dragapult ex created | teachers | v2 |
|---|---:|---:|
| born Active **without** Fire+Psychic | **0.058** | **0.220** |
| born Active with Fire+Psychic | 0.308 | 0.293 |

Split by result, because a losing player is in bad spots by definition:

| born Active unpowered, per game | wins | losses |
|---|---:|---:|
| teachers | 0.090 | 0.161 |
| v2 | 0.071 | **0.667** |

In games we *win* we match the teachers exactly (0.071 vs 0.090). In games we
lose we do it **4.1x** as often as the teachers do in games *they* lose. This
is the sharpest single contrast in the audit.

Caveat, and it matters for how to fix it: on held-out teacher states the agent
is *more* conservative than the teachers, not less —

| evolve target offered | offers | teacher takes | v2 takes |
|---|---:|---:|---:|
| Active, already powered | 130 | 0.385 | 0.369 |
| Active, unpowered | 194 | 0.273 | **0.170** |
| bench, powered | 189 | 0.328 | 0.249 |
| bench, unpowered | 1,071 | 0.106 | 0.071 |

So this is a **state-distribution** effect, not a preference the model learned:
the live states in which we evolve an unpowered Active are states the teachers
almost never reach, and the model has no training signal there. A learned fix
will not reliably bind; a hard precondition will.

## 5. Matchups: the deficit is two cells, and the field we met was easy

Applying the teachers' per-archetype win rate to the archetypes v2 actually
faced:

| opponent | v2 n | teacher n | teacher wr | v2 wr | expected w | actual w | residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mega Lucario ex | 6 | 87 | 0.655 | 0.333 | 3.93 | 2 | **−1.93** |
| Dragapult ex (mirror) | 3 | 322 | 0.571 | 0.000 | 1.71 | 0 | **−1.71** |
| Teal Mask Ogerpon ex | 2 | 42 | 0.786 | 0.500 | 1.57 | 1 | −0.57 |
| Cinderace | 1 | 18 | 0.889 | 0.000 | 0.89 | 0 | −0.89 |
| Alakazam | 4 | 202 | 0.767 | 0.750 | 3.07 | 3 | −0.07 |
| Marnie's Grimmsnarl ex | 4 | 191 | 0.749 | 0.750 | 2.99 | 3 | +0.01 |
| Mega Kangaskhan ex | 3 | 68 | 0.500 | 0.667 | 1.50 | 2 | +0.50 |
| Crustle | 2 | 5 | 0.600 | 1.000 | 1.20 | 2 | +0.80 |

**26 games, 14 wins, 17.87 expected. Residual −3.87.** The field-adjusted
expectation is 0.687 — we met an *easier* mix than the teachers' own 0.651 —
and we returned 0.538. Mega Lucario and the mirror are −3.64 of the −3.87;
every other cell with real teacher evidence is at or above expectation.

Three of the eight tail games are Mega Lucario. Mechanically it is a real
squeeze: Mega Lucario ex has 340 HP against Phantom Dive's 200, so it never
dies in one hit, while Mega Brave does 270 into our 320 and Aura Jab
re-accelerates from the discard. But the teachers win it 0.655 over 87 games,
so it is not a lost matchup. Six games measures nothing on its own; it is
worth attention because it is the largest cell *and* it dominates the tail.

## 6. Routing: the OOD gate is real, smaller than reported, and not worth fixing

| | teacher held-out (16,380) | v2 live (2,521) |
|---|---:|---:|
| ranker used | 0.7998 | 0.7497 |
| unrouted | 0.1647 | 0.1722 |
| optional fallback | 0.0910 | 0.0877 |
| **out-of-distribution fallback** | **0.0004** | **0.0233** |
| single-semantic fallback | 0.0351 | 0.0555 |
| feature / score errors | 0 | 0 |

It still fires almost entirely in one place — 54 of 402 Phantom Dive counter
placements (13.4%) and 3 of 58 card searches — triggered by opponent Pokemon
the corpus never contained (Great Tusk 51, Chewtle 20, Drednaw 13, Eevee 6).
Revision 1 measured 22% of placements on 12 games; on 26 it is 13.4%, and the
rate is falling as v2 climbs into a more meta field, exactly as predicted.

**The proposed fix was then measured and is worth nothing.** Relaxing
`_supported` so that opponent-owned candidates skip the `candidate_card_id`
identity check hands all 54 placements back to the model. The model then makes
the *same* placement as the rule policy on 47 of 54 (87.0%). The seven
disagreements are all in one game we won 6-0, and they look worse, not better:
the rule policy concentrates counters on one Crustle until it dies
(counters_to_ko 9 -> 8 -> 7 -> 6 -> 5 -> 4) while the model spreads onto a
fresh one at counters_to_ko 15.

Ranked #1 in revision 1. It is now closed: the hand-written placement rule is
already doing what the model would do.

## 7. Where the model still disagrees with the teachers

152 held-out episodes, 7,054 main-phase decisions, main-phase agreement 0.6425.
Largest deviations per decision offered: Crushing Hammer +0.252, Fezandipiti ex
ability +0.161, evolve Drakloak +0.149, against Spikemuth Gym −0.110, Poke Pad
−0.059, evolve Dragapult ex −0.057, Ultra Ball −0.046.

Read as ordering, not volume: per game in 80 arena games v2 plays 2.45 Crushing
Hammer against the teachers' 2.54. 14.3% of all decisions are
`same_turn_ordering` and the order-insensitive Top-1 is 0.8891 against the
strict 0.7462. The volume gaps that survive the arena's larger sample are
search and recovery — Ultra Ball 1.30 vs 1.64, Buddy-Buddy Poffin 1.13 vs 1.56,
Poke Pad 2.05 vs 2.35, Night Stretcher 0.80 vs 0.96 — against Crispin 1.64 vs
1.36 and Boss 1.13 vs 0.98. v2's new columns describe the Energy route in
detail and add nothing about board width, and the policy drifted accordingly.
Section 3 shows the same drift in the outcome that matters: 0.77 bench-built
Dragapult ex per game against 1.19.

## 8. The pinned teacher is worth about one point

Every identity swept on the exact per-team held-out split, 14,088 decisions:
best 16462035 at 0.7243 pooled / 0.6458 MAIN, current 16380946 at 0.7183 /
0.6359, worst 16422241 at 0.7055 / 0.6173. `teacher_team_id` is the 4th
highest-gain feature, so the pin changes behaviour more than it changes
agreement — the current pin has the cohort's highest Crushing Hammer rate
(0.511 against a pooled 0.360) — but as a measured lever it is +0.006 pooled.

## 9. Ranked v3 candidates

1. **Refuse to evolve into an Active Dragapult ex that cannot attack this
   turn.** A hard precondition, not a feature: decline the evolution when the
   target is the Active, it will not hold Fire+Psychic after every attachment
   available this turn, and it is not already lethal. Section 4 shows the
   learned policy is *more* conservative than the teachers on states the
   corpus covers, so this has to be a rule to bind where it matters. Sweep
   every legal index offline first and confirm it binds >0 times.
2. **Give the model board-width columns to match the Energy-route columns it
   got in v2**: route bodies that could still become a second attacker, line
   pieces left in deck, whether a search adds a body or a resource. Target the
   measured 1.577 -> 1.883 Dragapult ex per game and the 0.77 -> 1.19
   bench-built gap.
3. **Look at Mega Lucario specifically before touching anything global.** It
   is the largest matchup cell, carries 3 of 8 tail games and half the
   residual, and the teachers win it 0.655 over 87 games. Read their games,
   not ours: 6 games of ours is not a measurement, 87 of theirs is.
4. **Re-pin to 16462035** — free, +0.010 MAIN, and it moves the Crushing
   Hammer habit from a 0.511 pilot to a 0.453 one. Ship with 2 so the two are
   not confounded.
5. **Do not relax the OOD gate.** Measured in section 6: 87% agreement with
   the rule policy it would replace, and the disagreements look worse.
6. **Do not chase main-phase ordering.** 14.3% of decisions are pure ordering
   and the order-insensitive Top-1 is 0.889.
7. **Do not re-tune the other matchups.** Alakazam, Grimmsnarl and Kangaskhan
   are all at or above the teachers' rate.

## 10. What this still does not settle

* The mirror at 0-3 is the second-largest residual and three games cannot
  distinguish a real hole from noise. The teachers are 0.571 over 322.
* Mega Lucario at 2-4 has a 95% interval of roughly [0.04, 0.78] around 0.333,
  which contains the teachers' 0.655.
* Whether the unpowered-Active evolution *causes* losses or merely *marks*
  them is not settled by observation. The proposed fix is cheap and bounded
  either way, but the honest claim is a 4.1x contrast in matched conditions,
  not a causal estimate.
