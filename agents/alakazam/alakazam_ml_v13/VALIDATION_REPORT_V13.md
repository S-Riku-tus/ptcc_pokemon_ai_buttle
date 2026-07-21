# VALIDATION REPORT V13

## Static validation

- `deck.csv`: 60 cards
- ACE SPEC: Enriching Energy 1, other ACE SPEC 0
- Python compile: pass
- Runtime import with cg test stub: pass
- Distilled model load: pass
- ML live override default: disabled

## Tests

`pytest -q` result: **47 passed**

Covered regression states include:

- v11 deck to v13 deck exact one-card swap
- Rich Energy net hand delta +3
- Dunsparce immediate recycle priority
- Rich Energy not counted as Psychic fuel
- Rocket Articuno protects Basic Team Rocket Pokemon but not evolved Spidops
- Powerful Hand rejected into Mist/protection
- Boss rejects Grimmsnarl ex -> Morgrem lower-prize replacement
- Boss can escape protected Active to an unprotected Spidops KO
- Mist public-copy probability decays and releases after expected copies
- final Hammer reserve remains in high-Mist matchup after one copy
- Fezandipiti ex alternate-attacker mode under complete effect lock
- ML feature extraction for Rich/Mist states

## Not validated here

- Official `cg` engine full match
- Kaggle Validation Episode
- ladder rating
- full self-play versus v11/top agents

Therefore this is a statically and Golden-state validated candidate, not a measured 1000-rating agent.
