# grimmsnarl_ml_v16 — what the Shadow Bullet is worth

Evidence base: the 110 rated games of `grimmsnarl_ml_v15` across two
byte-identical submissions (`55404196`, `55409394`), 66-44, mean opponent
rating 886.4. All numbers below are recomputed from those replays by the
scripts named in each section, not carried over from a prior write-up.

## 1. v15's change worked, and closed its own question

| KPI | v14 | v15 | rank-3 pilot |
| --- | ---: | ---: | ---: |
| first Shadow Bullet, own turn | 3.60 | **2.84** | 2.96 |
| Shadow by own turn 2 | 19.0% | **40.9%** | 38.0% |
| Shadow by own turn 3 | — | **82.7%** | — |
| turn-ends with a fuelled Grimmsnarl ex unused | — | **4 / 187** | — |

The decisive new number is that in the losses the mean turn of the **first
Grimmsnarl ex** and the mean turn of the **first Shadow Bullet** are the same,
3.048 and 3.048. There is no gap left between owning an attacker and using it.
Every remaining late Shadow Bullet is a late Grimmsnarl ex.

`scripts/analyze_grimmsnarl_v16_prize_conversion.py`

## 2. Four candidate levers, measured and rejected

The v15 autopsy proposed a two-turn prize planner, a mirror Bench-30 / Boss /
Adrena-Brain target plan, and a first-attacker construction gate. All three
were tested at decision level against v15's own replays and all three are
already saturated.

| candidate | measurement | verdict |
| --- | --- | --- |
| two-turn Boss route | a playable Boss's Orders in hand would have added a prize on **1 of 446** Shadow Bullets | dead |
| Bench-30 targeting | an offered lethal Bench-30 was passed over **2 of 397** times | dead |
| Adrena-Brain | taken on **98.6%** of the 346 turns it was offered; **100%** in losses | dead |
| attacker construction gate | a Grimmsnarl-line play was passed over at END **1 of 176** no-attacker turns | dead |

Every one of these is offer-side, matching the pattern the project has hit
before: v15 plays essentially everything it is shown, so no preference,
planner or ranker change can move these. The Adrena-Brain gap in the mirror
(winner 7.61 uses a game, loser 3.81) is entirely availability — and
availability is already in our favour, with an energised Munkidori by own
turn 2 in 91.7% of mirrors against the opponents' 72.2%.

`analyze_grimmsnarl_v16_boss_routes.py`, `_ability_uptake.py`,
`_energy_allocation.py`, `_mirror_behaviour.py`

## 3. What is not saturated

### 3a. 84 swings that were provably worth zero

A Shadow Bullet is worth nothing when the opposing Active prevents all damage
from us *and* the Bench-30 can take no prize. That happened **84 times** over
the 110 games, concentrated in the 15 wall games:

| wall matchup | wins (7) | losses (8) |
| --- | ---: | ---: |
| Shadow Bullets a game | 6.00 | **8.25** |
| share taking no prize | 64.3% | **86.4%** |
| prizes per Shadow Bullet | 0.595 | **0.212** |
| stalled turns a game | 0.0 | **2.88** |
| deck left at the end | 10.9 | **7.6** |

Episode **91663479** is the failure in one game: Cornerstone Mask Ogerpon ex
Active, **an empty Bench for all 24 turns**, 21 Shadow Bullets, 0 prizes,
deck 0, lost by deck-out while holding two Boss's Orders with nothing to gust.
Episode 91770999 is the same shape at 10 swings.

**The route out has always been in the deck.** The three walls block different
things and none of them blocks Marnie's Morgrem:

| wall | blocks | Morgrem? |
| --- | --- | --- |
| Crustle, Sylveon | Pokémon **ex** | lands |
| Cornerstone Mask Ogerpon ex | Pokémon **with an Ability** | lands |
| Neutralization Zone | ex and V, onto non-Rule-Box | lands |

Morgrem is neither an ex nor an Ability holder, so Corkscrew Punch's 60 goes
through all of them. Against a 210 HP Ogerpon that is four swings, and 91663479
had twenty-five turns.

Only Morgrem and Impidimp qualify: Basic Darkness is the only Energy in the 60,
and Frost Smash is {W}{C}, Mind Bend is {P}{C} and Chilly is {W}, so Froslass,
Munkidori and Snorunt can never attack at all.

Opportunity, with the deck's real energy typing applied:

| | count |
| --- | ---: |
| provably dead swings | 84 |
| a breaker was in play | 44 |
| already fuelled | 30 |
| route finishes within 8 turns and before deck-out | 35 |
| **last breaker evolved into Grimmsnarl ex under the wall** | **8** |

Those 8 are in 7 of the 15 wall games including all three worst, and every one
already had a fuelled Grimmsnarl ex in play — which is why 17 of 91663479's 21
dead swings had no breaker left on the board at all.

### 3b. A teacher escalation that generalised too far

v6 handed the `evolve_froslass` class to pilot 16371703 because the pinned
teacher takes that evolve on 95.7% of its own turns and that pilot on 80.5%.
Off the mirror it works: 85.6% uptake over v15's 110 games. On mirror boards it
produces a rate no pilot in the corpus plays.

| | offering turns taken |
| --- | --- |
| v15, mirror | **6 / 20 (30%)** |
| mirror opponents, identical 60 cards | **12 / 12 (100%)** |
| v15, every other matchup | 85.6% |

Fisher exact, 6/20 against 12/12: **p = 0.000112**.

Replaying all 104 stored mirror decisions that offered the evolve through the
shipped v15 isolates the cause:

| escalation | would evolve |
| --- | ---: |
| on (shipped) | **4 / 104** |
| off (v5 behaviour) | **33 / 104** |

**Honest limit.** This is an imitation gap, not an outcome gap. Within our own
36 mirror games taking the evolve does not itself predict winning (4 of 15
offering turns in wins, 2 of 5 in losses). The argument for changing it is that
a 30% rate matches neither the pin nor the escalation pilot nor the field, and
the fallback is v5's behaviour on one matchup.

## 4. v16

Two changes, model and deck byte-identical, each with its own kill switch, on
disjoint sets of games.

1. **`wall_break.py`** — while a Shadow Bullet is provably worth zero, advance
   a route to the breaker instead of throwing it (attack with it, fuel it, or
   retreat into it), and do not spend the last breaker on a Grimmsnarl ex
   evolution while that wall is up. `GRIMMSNARL_WALL_BREAK_DISABLE=1`.
2. **The mirror escalation gate** in `ml_runtime.py` — the `froslass_evolve`
   class is not escalated when the public-information router reports a mirror.
   `GRIMMSNARL_ESCALATION_MIRROR=on`.

### Deliberately not done

* **No forced Boss's Orders.** 1 missed Boss prize in 446 swings; the guard
  stands down whenever a playable Boss would take a prize.
* **No ban on the zero-damage Shadow Bullet.** Banning it outright took a
  previous wall specialist to 1-10. Both END and ATTACK close the turn, so the
  swing costs nothing on its own; it only matters that there was something
  better to do.
* **No stall circuit breaker.** The deck-out comes from the draw at the start
  of each turn, not from the swing, so refusing to attack saves no cards. What
  loses those games is having no route.
* **No two-turn prize planner.** Measured dead, see §2.

## 5. Measured footprint on v15's own boards

`scripts/probe_grimmsnarl_v16_footprint.py` replays every stored board through
v15 and v16 side by side, teacher-forcing the intra-turn history with the
action the game actually played.

**Counterfactual — 51 games (15 wall + 36 mirror), 5109 decisions, with
`GRIMMSNARL_WALL_BREAK_DISABLE=1` and `GRIMMSNARL_ESCALATION_MIRROR=on`:
0 differences.** v16 with both switches off is v15 decision for decision.

**Wall footprint — 22 differences over the same 1742 decisions, in 8 of 15 games:**

| replaced | with | n |
| --- | --- | ---: |
| Shadow Bullet | retreat (to the breaker) | 11 |
| energy attachment | energy attachment (redirected to the breaker) | 9 |
| Grimmsnarl ex evolve | retreat | 2 |

Guard counters over those games: 104 dead-swing turns / 523 dead-swing
decisions seen, and the guard stood down on most of them — `no_breaker_in_play`
151, `breaker_too_slow` 128, `breaker_would_be_sacrificed` 80,
`boss_prize_deferred` 46. It acted 22 times. `last_breaker_evolve_refused` 2
against `last_breaker_evolve_kept` 6: PRESERVE only fires when a route step
exists to spend the turn on instead, which is the conservative half of the 8
occurrences found in §3a.

**Mirror footprint — 45 differences over 36 games / 3367 decisions, in 11
games, and `wall_break` fires 0 times there** (no dead swing in any mirror
game), so the two changes really are disjoint and one ladder run attributes
both. 30 of the 45 land on the Froslass evolve itself; the rest are other
options inside the same select, which v6's `class` mode scores end to end as
the escalation pilot.

**Not covered by this probe:** the promotion after a forced retreat. The
replay continues with v15's actual action, so the retreat never resolves and
the promotion select never appears. That path is covered by
`tests/test_v16_wall_break.py` only.

### Live safety

`scripts/local_arena.py`, 50 games with the guard active: **0 crashes, 0
illegal selects**, 74 ms a move. 20 games against v15 went 13-7 and 30 against
a Crustle deck went 27-3, matching v15's 27-3 — neither is evidence of
strength. The arena cannot be paired (`--seed` does not seed the native
shuffle), and the baseline piloting Crustle does not build a real wall, so
these are smoke tests. The ladder is the test.

`agents/grimmsnarl/grimmsnarl_ml_v16/tests/`: 253 tests, including the
byte-identity assertion over every file v16 claims is unchanged from v15.

## 6. Promotion gate

| KPI | target |
| --- | --- |
| crash / illegal action | 0 |
| rating drift from 1000+, ≥ 50 games | > 0 (not a peak) |
| first Shadow Bullet, own turn | ≤ 3.1 (regression gate on v15's gain) |
| Shadow by own turn 2 | ≥ 0.35 |
| games lost to deck-out with a breaker in play | 0 |
| prizes per Shadow Bullet, wall matchup | > 0.212 |
| Froslass uptake when offered, mirror | > 0.8 |
| mirror rating drift | ≥ 0 |

Peak rating is explicitly not a promotion signal. The two changes fire on
disjoint game sets, so one ladder run attributes both.

## 7. Next, still unmeasured

* **Punk Up's multi-pick energy distribution.** It bypasses the ranker
  entirely and reaches "your Marnie's Pokémon", which includes Morgrem — so it
  could fuel a wall breaker for free. No metric scores it today.
* **The mirror prize race after the first Shadow Bullet.** We are faster (2.47
  against 2.89), take more prizes (3.81 against 2.56) and still go 21-15. None
  of the decision-level probes in §2 explains it.
* **Festival/Dipplin: 0-4**, mean opponent rating 1085, 0.5 Shadow Bullets a
  game. Four games is not a diagnosis, but it is the archetype that appears
  when the rating goes up.
* **Catastrophic losses to sub-800 opponents** as a KPI of their own.
