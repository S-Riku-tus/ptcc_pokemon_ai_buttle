# Alakazam ML v6 Validation Report

## Result

- Python compile: PASS
- Test suite: **22 passed**
- Deck count: **60**
- Boss's Orders: **3**
- Basic Psychic Energy: **2**
- Genesect: **0**
- Lucky Helmet: **0**
- Dunsparce / Dudunsparce: **3 / 3**
- ACE SPEC: Maximum Rod 1
- Authoritative fallback: `fallback_v3.py`
- ML mode: shadow-only by default
- Boss / Ability: RULE_ONLY

## Regression coverage

- only-body Active Dudunsparce cannot activate Run Away Draw
- Active Dudunsparce can still cycle when a ready Bench body remains
- Team Rocket's Articuno is a qualified high-value Boss target
- a disposable low-value Basic is not a Boss target
- Boss is rejected when the one-card hand spend loses the target KO
- Boss does not replace a better Active KO
- Boss never gives up a non-Boss immediate winning Active KO
- deck and metadata invariants
- ML guard and shadow-runtime invariants

## Hashes

- ranker_model.json before: `7b149ebd5ee3fd08ff9d8d692c76934a655c59081ab2be25ddcf2ca2132e5364`
- ranker_model.json after: `7b149ebd5ee3fd08ff9d8d692c76934a655c59081ab2be25ddcf2ca2132e5364`
- model unchanged: **yes**
- deck.csv: `d08565ebf564dfff70fb0131a3fd0e3161039a559ad6f20e01495ae017ca57af`

## Limitation

The uploaded agent archive does not contain the official `cg/` engine package. The included v6 tests use a minimal API-compatible stub for the newly changed authoritative logic. Full engine self-play and Kaggle Validation Episode remain external checks.
