# Submission 54797424 and dual-leader log analysis

## Scope and data quality

- Newly supplied archive: submission `54797424`
- Manifest rows: 750
- Replay files present: 748
- Completed, reward-bearing replays used: **747**
- Earlier reference submission `54837973`: **161** replays
- Combined evidence base: **908** replays
- Both pilots used exactly the same 60-card list as this agent.

The two pilots are independent leaderboard agents:

- `54837973`: `__Taichicchi__`
- `54797424`: `bono`

The new sample is much larger, but raw win rates are not directly comparable because opponent distribution and evaluation timing differ.

## Results of submission 54797424

| Metric | Result |
|---|---:|
| Record | **455–292** |
| Win rate | **60.9%** |
| Going first | 261–136, **65.7%** |
| Going second | 194–156, **55.4%** |
| Mean own turns | 6.29 |
| Mean attack turns | 2.09 |
| Attack-turn rate | 33.3% |
| Mean first attack | 3.01 own turns |
| Mean first Shadow Bullet | 3.50 own turns |

Going first remained clearly superior. Across both leader samples, going first was 330–164 (66.8%), while going second was 233–181 (56.3%). The policy therefore remains `go_first = True`.

## Matchups in the new sample

| Opposing archetype | Games | Record | Win rate |
|---|---:|---:|---:|
| Alakazam / Dudunsparce | 301 | 180–121 | 59.8% |
| Marnie's Grimmsnarl / Froslass | 132 | 81–51 | 61.4% |
| Team Rocket's Spidops | 67 | 37–30 | 55.2% |
| Crustle / Mega Kangaskhan | 67 | 50–17 | 74.6% |
| Dragapult ex | 41 | 21–20 | 51.2% |
| Cynthia's Garchomp | 30 | 14–16 | 46.7% |
| Mega Lucario ex | 26 | 13–13 | 50.0% |
| Mega Starmie / Cinderace | 13 | 9–4 | 69.2% |
| Festival Dipplin | 11 | 6–5 | 54.5% |

The largest improvement opportunities remain Cynthia's Garchomp, Dragapult, Spidops, and Mega Lucario. Crustle is not a losing matchup for the leader, but it requires a different prize route from normal Shadow Bullet damage.

## Identical 60-card construction

Both upper-ladder pilots used the same list:

- Darkness Energy 10
- Munkidori 4
- Marnie's Impidimp 4
- Marnie's Morgrem 3
- Marnie's Grimmsnarl ex 3
- Froslass 2 / Snorunt 2
- Rare Candy 3
- Unfair Stamp 1
- Buddy-Buddy Poffin 4
- Night Stretcher 3
- Poké Pad 4
- Handheld Fan 2
- Boss's Orders 2
- Team Rocket's Petrel 4
- Lillie's Determination 4
- Dawn 1
- Spikemuth Gym 4

No deck-list change is justified before testing the logic-only revision.

## Stable decisions shared by both pilots

### Punk Up uses essentially all five Energy

Combined Punk Up target selections:

| Target | Selections | Share |
|---|---:|---:|
| Marnie's Grimmsnarl ex | 3,661 | 55.9% |
| Marnie's Impidimp | 2,413 | 36.8% |
| Marnie's Morgrem | 476 | 7.3% |

There were 6,550 attachment target selections across 1,329 Energy-search events, or **4.93 Energy per Punk Up**.

The correct objective remains:

1. Make the current Grimmsnarl attack-ready.
2. Put two Energy on a distinct evolution body with a visible route.
3. Only then use spare Energy as a retreat or denial buffer.

The new log shows that the preferred backup is not automatically Morgrem. Impidimp is often better when Rare Candy plus Grimmsnarl is already visible.

### Munkidori is the main manual Energy target

Across both pilots, manual Darkness Energy targets were:

| Target | Attachments | Share |
|---|---:|---:|
| Munkidori | 1,525 | **27.7%** |
| Marnie's Impidimp | 1,430 | 25.9% |
| Marnie's Grimmsnarl ex | 909 | 16.5% |
| Snorunt | 627 | 11.4% |
| Froslass | 524 | 9.5% |
| Marnie's Morgrem | 499 | 9.0% |

v2 only gave Munkidori a high score after Shadow Bullet was already live or the backup was already close. That is too late. Once a concrete Grimmsnarl route is visible, the manual attachment can activate Munkidori because Punk Up normally supplies the attacker later.

### Search hierarchies are highly consistent

Combined selection shares:

**Poké Pad**

- Munkidori: 31.9%
- Morgrem: 25.6%
- Impidimp: 17.2%
- Froslass: 16.5%
- Snorunt: 8.8%

**Spikemuth Gym**

- Grimmsnarl ex: 38.9%
- Morgrem: 35.8%
- Impidimp: 25.3%

**Buddy-Buddy Poffin**

- Impidimp: 65.6%
- Snorunt: 34.4%

**Night Stretcher**

- Darkness Energy: 38.7%
- Impidimp: 19.1%
- Grimmsnarl ex: 12.7%
- Morgrem: 12.1%
- Munkidori: 7.2%

These distributions support v2's role-based search, but the larger sample exposes one missing condition: if Impidimp, Rare Candy, and Grimmsnarl are already visible, Morgrem is not a route deficit and Poké Pad should not search it merely because no Morgrem is in hand.

## What should not be copied from the new pilot

The new pilot used basic and Stage 1 attacks much more often:

| Attack | Wins / game | Losses / game |
|---|---:|---:|
| Shadow Bullet | **2.14** | 1.82 |
| Filch | 0.45 | **0.76** |
| Impidimp Punch | 0.44 | **0.74** |
| Morgrem Punch | 0.22 | **0.37** |

The weaker attacks occur more frequently in losses. They are mainly evidence that Grimmsnarl was not completed, not evidence that final v2 should promote chip attacks over development or Shadow Bullet. final v2 therefore preserves the existing attack hierarchy.

## Board depth

New-pilot board counts by result:

| Own turn | Wins | Losses |
|---|---:|---:|
| T1 | 3.70 | 3.58 |
| T2 | 4.85 | 4.49 |
| T3 | 5.25 | 4.86 |
| T4 | 5.43 | 4.86 |

A developed board remains beneficial. However, the new pilot's overall board was slightly leaner than the earlier pilot's. The implication is not to reduce board depth globally; it is to stop adding redundant evolution or Froslass pieces once Shadow Bullet and one real backup are already secured.

## Crustle correction

Crustle prevents attack damage from opposing Pokémon ex. Therefore Grimmsnarl ex's 180 Active damage is zero against an Active Crustle, although the attack can still apply its separate Bench-30 route when legal.

In the 67 new Crustle games:

- Record: 50–17
- Adrena-Brain target selections included Crustle 314 times and Dwebble 103 times.
- Boss targets included Crustle 18 times, Dwebble 8 times, and Mega Kangaskhan ex 7 times.

The winning route is damage-counter placement and selective gusting, not treating Crustle as a normal 180-damage KO. v2's `shadow_damage()` did not model this and could incorrectly reject Boss because it believed the Active was already KO-able.

## Changes selected for final v2

1. Model Crustle's Pokémon-ex damage prevention.
2. Lower Shadow Bullet's main score when its Active damage is zero, while retaining the Bench-30 route.
3. Allow Boss to target a real KO while Crustle is Active.
4. Recognize a complete Impidimp + Rare Candy + Grimmsnarl route and stop redundant Morgrem searches.
5. Make Punk Up choose the backup body with the visible evolution route rather than always preferring Stage 1.
6. Activate Munkidori earlier when a concrete Grimmsnarl route exists.
7. Reserve a live Shadow Bullet over Froslass or a third completed evolution line.
8. Add high-frequency target families from the larger sample: Dunsparce, Dwebble, Drakloak, mirror evolution bodies, Crustle, Dragapult, and Mega Kangaskhan.
9. Prefer removing damage counters from a low-HP powered Munkidori when that prevents losing the damage-relocation engine.

## Changes deliberately not made

- No card-count changes.
- No switch to going second.
- No blanket increase to Filch, Impidimp Punch, or Morgrem Punch.
- No forced five- or six-body board target.
- No unconditional Boss usage.
- No change to the distinction between Shaymin's attack-damage protection and Battle Coliseum's damage-counter protection.
