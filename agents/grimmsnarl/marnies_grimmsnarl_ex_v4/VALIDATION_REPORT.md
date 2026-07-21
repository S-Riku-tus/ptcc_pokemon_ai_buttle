# Validation Report — Marnie’s Grimmsnarl ex v4

## Scope

Static/runtime-compatible validation only. The official competition engine and external matchup harness were not available, so no claim is made about live win rate.

## Results

- Deck count: **60**
- Deck changed from v3: **No**
- Python compile: **Pass**
- Self-contained imports: **Pass**
- Golden-state tests: **22/22 pass**
- Duplicate policy method names: **None**
- Submission payload: **main.py, policy_base.py, deck.csv only**
- Cache/bytecode in archives: **None**

## v4-specific guarantees

1. Spikemuth Gym does not request Morgrem when the Candy route is already complete.
2. A missing entire Grimmsnarl line triggers Pokémon search.
3. Unfair Stamp protects a large unique backup hand.
4. Unfair Stamp precedes the attack when it refills a small hand against a large hand.
5. A useful second Munkidori can be played before Shadow Bullet.
6. A second Munkidori is blocked when it consumes a reserved line slot.
7. Rare Candy completing the backup stays above the live attack.
8. A reachable two-prize short route outranks isolated one-prize chip damage.

## Remaining uncertainty

Live engine validation and matchup evaluation must be performed by the user. The recommended comparison is v3 vs v4 with identical seeds and seat swaps, tracking attack continuity, Stamp quality, second-Munkidori activation, and prizes created by Bench-30/Adrena-Brain combinations.
