# v11 changelog

## Evidence and diagnosis

- v10 public result: 26-33 over 59 games (44.1%), ending near rating 815.
- In 11 openings containing both Abra and Dunsparce, v10 chose Abra all 11
  times and went 3-8 in that subset.
- The saved eight-team Alakazam study found a 54-card common core; this favors
  sequencing fixes before a deck rewrite.
- Broad backup/runway rules lacked replay support, so they were removed.

## Policy

- Added Dunsparce-over-Abra opening choice.
- Added same-turn and immediate-KO Rare Candy projection, Rare Candy target
  selection, and Kadabra-to-Candy expected-value routing.
- Added probability- and pressure-gated low-deck Psychic Energy digging.
- Narrowed Fezandipiti energy, Shaymin, and non-mirror Xerosic decisions.
- Retained v10 Boss, Hammer/Mist, support pivot, and attack authority.

## ML

- Added probability, evolution-draw, Candy-tempo, and setup interaction
  features.
- Kept the existing ranker shadow-only and unchanged; no unsupported live
  authority expansion was made.

## Deck

- No change from v10. Keep Max Rod, Shaymin, and six Psychic-providing Energy.
