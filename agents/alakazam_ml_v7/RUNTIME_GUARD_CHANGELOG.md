# Runtime guard change log — v4 logic audit 0.3.0

## Preserved deck revision

Dunsparce 3, Dudunsparce 3, Psyduck 0, Basic Psychic Energy 3, total 60.

## Deterministic fallback fixes

1. Evolution hand-size calculation now includes Kadabra/Alakazam draw and Rare Candy's complete effect.
2. Search effects use exact deck cost and net hand change.
3. A search is considered a secured backup only if the hypothetical result reduces `backup_eta` to 1 or less.
4. Search lethal credit is awarded only when that exact search creates the KO.
5. Fezandipiti ex reserves core Bench capacity and is no longer proactively fetched.
6. Fezandipiti draw is stopped when current KO and backup are already secure or the deck race is unsafe.
7. Genesect cannot take the last Fez/core slot and still requires an immediately attachable Lucky Helmet.
8. Nighttime Mine requires an immediate tax effect against the opposing Active Tera attacker.
9. Survival Bench choices are role-aware instead of treating all Basics equally.
10. Optional role spends cannot turn a two-hit Powerful Hand line into a three-hit line.
11. Reachable-hand estimation no longer counts an unusable Dawn or Hilda.

## ML guard fixes

- Preserve the fallback-selected Bench role/card.
- Preserve the fallback-selected evolution card/stage.
- Preserve fallback attack and current-KO protections.
- Runtime scope identifier: `guarded_intent_preserving_v2`.

## Unchanged

- `ranker_model.json`
- confidence and margin thresholds
- model tree structure
- legal-action fallback
- deckout/last-body Dudunsparce safety
