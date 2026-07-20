# alakazam741_v7

## Base

v7 is a reset from the v3 attack-first Alakazam 741 plan, not an incremental
v6 branch expansion. It keeps the v3 pressure core and ports only confirmed
safety guards:

- prevent last-board Dudunsparce from using Run Away Draw into an empty board
- recognize Mist/Rock-style effect prevention before Powerful Hand
- avoid 0-damage Powerful Hand loops and optional draw while locked
- preserve deck floors and winning low-deck states
- preserve current KO math when a hand-spending action would lose the KO
- keep type-aware energy checks and over-attachment prevention
- keep legal fallback behavior on malformed observations

## Deck Change From v3

- Rich Energy 13: 1 -> 0
- Hyper Aroma 1082: 0 -> 1
- Dunsparce 305: 4 -> 3
- Lillie's Determination 1227: 0 -> 1

The deck remains 60 cards. No Mega Diancie ex, Dudunsparce ex, Alakazam 245,
Shaymin 343, or other v5/v6 technology package is included in this initial
version.

## Action Model

Each MAIN decision is classified into exactly one phase:

- SETUP: no ready Alakazam attacker exists
- PRESSURE: a ready Alakazam attacker exists
- RECOVER: the line was disrupted and must be rebuilt
- LOCKED: Powerful Hand is blanked by effect prevention
- ENDGAME: deck or prize state makes closing the game more important

Scores are wrapped in priority tiers:

- Tier 0: block illegal/self-destructive/deck-out actions
- Tier 1: win now or prevent a forced loss
- Tier 2: take a KO or make a meaningful attack
- Tier 3: create the attacking Alakazam
- Tier 4: prepare the next Alakazam
- Tier 5: search or refill needed cards
- Tier 6: disruption and extra development
- Tier 7: end the turn

The tier wrapper makes a meaningful attack beat routine draw or development in
PRESSURE. It still allows confirmed unlock actions, such as Enhanced Hammer on
Mist Energy, before attacking.

## Lillie's Determination

Lillie is a recovery card, not a generic high-score supporter. v7 uses it only
when the hand is thin and the board cannot yet make an attacker, or after a
small-hand disruption state. It is blocked when the current hand already has a
complete Abra-to-Alakazam route, when the current attack wins, or when spending
the hand would lower Powerful Hand below the needed KO math.

## Hyper Aroma

Hyper Aroma is the ACE SPEC. Search selection prioritizes Kadabra when the
Alakazam line is not online, then Dudunsparce when the draw engine is missing.
It is also gated by the same deck-floor checks used for other optional searches.

## Responsibility Split

v7 inherits BasePolicy for selection normalization, fallback, common dispatch,
board counters, card access, attackability, type-aware energy checks,
over-attachment prevention, and prize helpers. The Alakazam file keeps only the
deck-specific work: Powerful Hand damage, hand-size preservation, evolution
route scoring, Run Away Draw safety, mirror Xerosic timing, effect-lock handling,
and Alakazam-specific search order.

## Tests

`tests/test_alakazam741_v7.py` fixes the following golden states:

- deck composition and static validation
- PRESSURE attacks over optional draw and END
- last-active Dudunsparce does not Run Away Draw
- effect-prevented Powerful Hand is not selected
- locked boards do not take optional draw
- Enhanced Hammer unlocks Mist Energy before attack
- current KO is not lost by hand-spending actions
- attackable Alakazam does not retreat
- unready active retreats to a ready benched Alakazam
- Lillie does not break a complete route
- Lillie is used with a thin no-attacker hand
- Hyper Aroma selects Kadabra or Dudunsparce according to board need
- malformed observations still return a legal fallback
