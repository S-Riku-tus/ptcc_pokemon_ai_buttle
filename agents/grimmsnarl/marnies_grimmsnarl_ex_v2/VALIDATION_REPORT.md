# Validation report — Marnie's Grimmsnarl ex v2

## Completed checks

- `deck.csv`: exactly 60 cards.
- Deck SHA-256 is identical to v1; final v2 is a logic-only change.
- `main.py` and `policy_base.py`: Python syntax compilation passed.
- Runtime import passed using the fake `cg.api` Golden-state harness.
- Golden-state tests: **14 passed**.
- Duplicate class-method scan: passed.
- All `C.<constant>` references resolve to defined constants.
- Runtime is self-contained: `policy_base.py` is bundled beside `main.py`.
- Submission archive contains only `main.py`, `policy_base.py`, and `deck.csv` at the archive root.
- Full and submission ZIP integrity tests passed.
- `__pycache__` and `.pyc` files were removed before packaging.

## Golden states

1. Deck identity and key counts remain fixed.
2. Runtime contains its own `policy_base.py`.
3. The only Impidimp is protected during opening Active selection.
4. Punk Up completes a one-Energy backup before adding a third Energy elsewhere.
5. Normal search falls below a live Shadow Bullet.
6. Night Stretcher prioritizes Darkness Energy for an unpowered Munkidori.
7. Boss does not replace an equal Active KO.
8. A complete Impidimp–Rare Candy–Grimmsnarl route does not request Morgrem.
9. Punk Up prefers the visible Candy route over an unready Morgrem route.
10. Munkidori can be powered before Shadow Bullet is live when a concrete route exists.
11. Crustle takes zero Active attack damage from Shadow Bullet and exposes a profitable Boss route.
12. Froslass and third-line evolution do not replace a reserved Shadow Bullet.
13. Adrena-Brain source selection saves a critical low-HP Munkidori.
14. High-frequency route pieces from the 747-game sample receive target bonuses.

## Not completed in this environment

The official competition `cg` runtime and match harness were not available. Actual Validation Episodes, timing measurements, v1–v2 seat-swapped self-play, and Kaggle rating confirmation remain necessary before declaring final v2 stronger in live play.
