# Marnie’s Grimmsnarl ex v5

Rule-based Kaggle agent. v5 keeps v4's route-aware sequencing and adds two
Pokémon-TCG fundamentals from the 658.1-rating ladder log:

1. **Damage-immune walls** (Crustle / Sylveon / Neutralization Zone): stop
   hammering Shadow Bullet for 0, keep developing, and use Boss's Orders to gust
   a real target into the Active.
2. **Safe evolution placement**: build the next attacker on the Bench instead of
   feeding the Active into a Pokémon it cannot damage or a KO it cannot survive
   (Morgrem in particular, which has no Punk Up).

## Submission payload

- `main.py`
- `policy_base.py`
- `deck.csv`

The full package also contains strategy, changelog, metadata, tests, and validation notes.
