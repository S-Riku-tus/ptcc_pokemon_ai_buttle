# Grimmsnarl ML v15: the attack-access invariant

## Decision

v15 is v14 plus **one** module, `attack_access.py`, and nothing else. The 60
cards, the ranker, the fallback policy, the planner, the Petrel/dead-Stamp
residual, the wall safety gate and the telemetry router are byte-identical to
v14 (and therefore still identical to v8 where v14 was).

The invariant it adds is a single sentence:

> A turn never passes with a Shadow Bullet we could have reached.

`GRIMMSNARL_ROUTE_DISABLE=1` restores v14 exactly, which is how the change is
measured.

## Why this and not the rest of the autopsy

The v14 ladder autopsy is directionally right that the deficit is policy, not
deck — the same 60 cards are rated 1151.0, 1116.3 and 1113.7 on the current
board — but its first ranking of causes needed correcting, and the corrected
ranking has one item at the top.

v14 does **not** have a board-building problem:

| measurement | v14 | rank-3 pilot |
| --- | ---: | ---: |
| Grimmsnarl ex evolutions / game | 2.21 | 1.95 |
| Boss's Orders -> attack same turn | 79% | 82% |
| Boss's Orders -> prize same turn | 63% | 57% |
| Unfair Stamp -> attack same turn | 90% | — |
| Petrel -> attack same turn | 74% | — |
| non-Shadow turns after the first Shadow (losses) | 0.27 | 1.20 |

It has a *starting* problem, and only in the states where the Active is not the
attacker:

| measurement | v14 | rank-3 pilot |
| --- | ---: | ---: |
| first Shadow Bullet turn | 3.60 | 2.96 |
| first Shadow Bullet turn, losses | 4.33 | 3.12 |
| Shadow Bullet by own turn 2 | 19.0% | 38.0% |
| first Shadow, opening Snorunt | 4.11 | 3.13 |
| first Shadow, opening Impidimp | 3.52 | 2.87 |
| escape attachment onto the opening Snorunt | 0.33 | 0.63 |
| first switch turn, opening Snorunt | 5.0 | 1.61 |
| prizes per Grimmsnarl ex evolved (losses) | 0.77 | 1.04 |

Episode **91548124** is the mechanism in one game: two Grimmsnarl ex finished on
our second turn, the opening Snorunt still Active for the rest of the game, ten
Basic Darkness spread over Grimmsnarl (3), Impidimp (3) and Munkidori (4) and
**none** on the Snorunt, first Shadow Bullet on turn 15.

### The distinction nothing in v8 draws

`ready_grimms()` answers *can this Grimmsnarl ex pay Shadow Bullet*. Nothing
answers *can it be in the Active spot this turn*. `_active_can_retreat()` reads
only the Energy already attached, so

```
Active   Snorunt, 0 Energy
Hand     Basic Darkness
Manual attachment  unused
Bench    Grimmsnarl ex with 2 Energy
```

reads as "no attack this turn" when the truth is
`attach -> retreat -> promote -> Shadow Bullet`. Every non-Grimmsnarl body this
deck plays retreats for exactly **one** Energy (Impidimp, Morgrem, Snorunt,
Froslass, Munkidori = 1; Grimmsnarl ex = 2 — asserted against the card database
in the tests), so that route is always one attachment long, and the resource it
needs is the once-a-turn manual attachment — the one resource every other play
also wants. Losing it is irreversible for the turn, which is why no amount of
later scoring re-opens the route.

### Why a hard invariant is the *conservative* choice here

v8's rule policy already scores both steps above everything else it can do:

| v8 rule | score | condition |
| --- | ---: | --- |
| `score_attach` | 990,000 | manual attachment onto a non-Grimmsnarl Active while a ready Grimmsnarl is benched |
| `score_retreat` | 995,000 | a ready Grimmsnarl is benched and the Active cannot Shadow Bullet |

Those are the two highest non-lethal scores in the rule set, and MAIN is decided
by the ranker, so in v14 they are unreachable. v15 adds no strategy: it makes
the ranker respect v8's own top-priority pair, on a **strict subset** of the
boards where v8 would apply it, because it also requires the route to complete
this turn and the Shadow Bullet to be worth making.

## What the module does

Four teeth, and the set is closed: within a turn MAIN repeats until an attack or
an END, so a non-terminal action cannot *delay* an attack — it can only spend a
card. A turn can therefore fail to attack in exactly two ways.

| tooth | fires when | forces |
| --- | --- | --- |
| ACCESS | a ready Grimmsnarl is unreachable and the route completes this turn | the escape attachment, or the retreat it paid for, or fuelling an Active Grimmsnarl one Energy short |
| CONVERT | the turn would END with a worthwhile Shadow Bullet legal | the Shadow Bullet |
| BRIDGE | the turn would END while v8's own scoring would attack | v8's attack (Filch / Corkscrew Punch) |
| PROMOTE | a promotion select follows our own retreat | the ready Grimmsnarl |

Refusals are as important as the firings:

* **the wall is untouched.** The route's "worth" gate is the wall guard's own
  gate — real damage to their Active, or a Bench-30 that takes a prize now — so
  a damage-immune Active with nothing to snipe keeps v14's behaviour exactly: no
  forced retreat into a wall, no forced zero-damage Shadow Bullet. The rank-3
  pilot plays about 2.8 zero-active Shadow Bullets a game in wall matchups, so
  banning them would have been a mistake.
* **no route is invented.** Debt of two Energy, the manual attachment already
  spent, an already-used retreat, a sleeping or paralysed Active, or no Darkness
  in hand all mean "no route", and the caller's index stands.
* **an Active-side route wins.** If the Active can evolve into Grimmsnarl ex, or
  the Rare Candy in hand does it, that is preferred to the escape route: it
  needs no Darkness and no retreat, and Punk Up fuels the fresh body.
* **stickiness without a stored plan.** The route is recomputed from the board
  every decision; paying the escape debt is precisely what turns the next step
  from "attach" into "retreat". v12 failed by re-searching every micro-decision
  — the answer here is a plan that cannot go stale, not a longer search.
* **multi-pick selects are never touched**, so Punk Up and Poffin budgets are
  bit-identical to v8.

## Same-board footprint

`scripts/probe_grimmsnarl_v15_route.py` drives a game with v14 and asks v15 for
an answer on every identical board (`footprint_v15_vs_v14.json`, 8 games).

| policy | differences from v14 | rate |
| --- | ---: | ---: |
| v15, `GRIMMSNARL_ROUTE_DISABLE=1` | **0 / 1,686** | 0.00% |
| v15 | **9 / 1,559** | **0.58%** |
| (v13, for scale) | 514 / 2,450 | 20.98% |
| (v14, for scale) | 10 / 2,450 | 0.41% |

Every one of the nine is the route and nothing else:

| v14 played | v15 plays | n |
| --- | --- | ---: |
| play a card | energy attachment | 2 |
| energy attachment (elsewhere) | energy attachment (Active) | 1 |
| evolve | retreat | 2 |
| evolve | energy attachment | 1 |
| ability | retreat | 3 |

which is exactly `escape_attach_forced = 4` plus `retreat_forced = 5`. The
trapped state itself occurred on 10 turns in those 8 games, every one of them
with a worthwhile Shadow Bullet at the end of the route.

## Golden states

`agents/grimmsnarl/grimmsnarl_ml_v15/tests/test_v15_attack_access.py` (26 cases)
pins the states the autopsy asked for, including the refusals:

1. Active Snorunt 0E + Darkness in hand + ready Bench Grimmsnarl -> the escape
   attachment is forced over optional setup, and over attaching to the Bench
   attacker instead.
2. the escape paid -> the retreat is forced, and the promotion select that
   follows takes the ready Grimmsnarl over a Snorunt.
3. Active Grimmsnarl one Energy short -> fuel it rather than retreat.
4. a non-Shadow attack loses to an open Shadow route.
5. manual attachment spent / already retreated / asleep / paralysed / no
   Darkness / two-Energy debt -> **no route invented**.
6. an Active Grimmsnarl-ex evolve, and the Rare Candy that does it, are
   preferred to the escape route.
7. a valueless wall -> the route does not open and the wall guard's END stands;
   a wall with a Bench-30 prize -> the route does open.
8. a turn never ends with a worthwhile Shadow Bullet unspent, but setup before
   that Shadow Bullet is still allowed.
9. the bridge attack is v8's judgement, not ours: v8's END is kept.
10. multi-pick selects untouched; promotion never fires outside our own route; a
    new game in a reused process drops pending state; every internal failure
    returns the caller's index.
11. the retreat-cost table matches the card database, and
    `files_identical_to_v14` really is identical (sha256).

## Deliberately not in this version

Each of these was in the autopsy and is being *declined* for a stated reason,
because the point of the version is that one change is measurable.

| candidate | why not now |
| --- | --- |
| general pre-Shadow setup gate (fewer Poké Pad / Poffin / Petrel before the first Shadow) | setup does not end the turn, so it cannot be what delays the attack; the measured counts are confounded by having more turns before the first Shadow. Inside the trapped state the route forcing already suppresses it. |
| `punk_search_budget` escape reserve | Punk Up draws from the deck and only reaches Marnie's Pokémon, so it can never pay a Snorunt or Munkidori escape. The current budget reproduces the elite band's own count exactly on 51.9% of activations; trading that for a second-order draw-probability effect is not worth it. |
| Boss's Orders / Unfair Stamp / Petrel changes | all at or above the rank-3 pilot's conversion rate on v14's own games. |
| Froslass suppression | identical 0.93 per game in v14's wins and losses. |
| banning zero-damage Shadow Bullet | top pilots play ~2.8 a game in wall matchups. |
| more Grimmsnarl lines / faster third body | v14 already evolves more than the rank-3 pilot, and *more* in losses than in wins. |
| two-turn prize planner | the next version's work, and it should be measured after the access defect is gone, or the two effects cannot be separated. |
| matchup experts, teacher averaging, DAgger | v9 and v13 are the evidence that generic models mislabelled as experts cost rating; a real expert needs matchup-specific trajectories. |
| any deck change | the same 60 cards reach 1151 in other hands, and changing deck and logic together destroys the attribution. |

## Promotion gate for the ladder run

The KPI list changes with this version.

* **primary:** first *Shadow Bullet* turn, and the share of games with one by our
  own turn 2. First *attack* turn is demoted — Filch and Corkscrew Punch count
  as attacks and hid 0.15 of a turn of the gap.
* **new, from the guard's own counters:** trapped turns per game, share of them
  that reach a Shadow Bullet, forced attachments, forced retreats,
  `ends_with_ready_attacker` (must stay 0).
* **rating:** drift >= 0 over at least 50 games started from 1000+, then 30 from
  1050+. Peak rating is **not** a promotion criterion — v13-b peaked at 1112.8
  and averaged -5.76 per game above 1050.
* **counterfactual:** `GRIMMSNARL_ROUTE_DISABLE=1` must keep reproducing v14
  decision for decision.

## Next, in order

1. Re-run the ladder telemetry against these counters. If `trapped_turns`
   decreases and the first-Shadow tail shrinks on real opponents, the access
   defect is closed.
2. Two-turn prize planner: Shadow Bullet's Bench-30 + Adrena-Brain + Froslass +
   Boss's Orders scored as one prize route rather than four decisions, with an
   **orphan damage** measurement (how much Bench-30 ever becomes a prize) as its
   KPI.
3. Distinct backup attacker ETA <= 1 measured at end of turn, not "an Impidimp
   exists".
4. Matchup residuals trained on real matchup trajectories, plus DAgger over our
   own failure states — the states above are ones top pilots rarely reach, so
   plain behaviour cloning has no data for them.
