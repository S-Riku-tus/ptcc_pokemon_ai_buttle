# v12 changelog

## Scope

v12 inherits v11's deck, model, features, and all unrelated deterministic
rules. This branch changes only Abra's attack-effect switch and Team Rocket's
Articuno handling.

## Teleport destination

- Distinguish Abra's `SWITCH` effect (`effect.id == 741`) from KO promotion,
  retreat, Boss's Orders, and other switch effects.
- Prefer Dunsparce, Dudunsparce, and a spare Abra over the evolution line.
- Estimate whether Fezandipiti ex survives the next opponent turn using visible
  attacks, current Energy, one possible next attachment, weakness/resistance,
  Alakazam hand damage, and a conservative powered-ex fallback.
- Score Kadabra and both Alakazam cards below every ordinary alternative. A
  mandatory single target is still selected legally.

## Team Rocket's Articuno

- Preserve the exact scope of global effect-protection abilities. Articuno now
  blocks Powerful Hand only on Basic Team Rocket Pokemon.
- Reject Powerful Hand against a protected target instead of valuing the
  attack as if it dealt damage.
- When the Active is protected, allow Boss's Orders to take a same-turn KO on
  an unprotected Team Rocket Evolution or non-Team-Rocket Bench Pokemon.
- When all visible targets are protected, allow gradual Energy investment in a
  damage attacker: Fezandipiti ex first, then Dudunsparce, Dunsparce, or Shaymin.
- Bring a ready breaker Active, keep an invested Dunsparce in front instead of
  using zero-damage Trading Places, and prioritize Articuno for Cruel Arrow.

## Unchanged

- `deck.csv` and `ranker_model.json`
- v11 evolution, Rare Candy, emergency Energy draw, Shaymin, Hammer, Xerosic,
  and ordinary Fezandipiti rules
- ML remains shadow-only by default

