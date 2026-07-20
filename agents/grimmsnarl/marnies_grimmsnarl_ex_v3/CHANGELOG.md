# Changelog

## v3.0.0

Logic-only change over v2 (deck and `policy_base.py` unchanged).  Driven by a
diagnosis of 1000+ head-to-head games against a strong hand-scaling Alakazam and
validated with a large-sample A/B (see `VALIDATION_REPORT.md`).

### Tempo / disruption

- **Unfair Stamp is now played as a free Item before the attack**, with priority
  scaled to the opponent's hand size.  v2 scored it below Shadow Bullet, so on a
  turn it could attack it attacked-and-ended and effectively never stamped.
  Stamp resets the opponent to two cards, starving hand-size damage/draw engines
  (Alakazam's *Powerful Hand* = 2 damage counters per card in hand).  This was the
  measured driver of the win-rate gain and it generalised across two opponents.
- `reserve_adjust` no longer caps Unfair Stamp below the attack (its own scoring
  now handles the ordering).

### Spread engine

- **Second Munkidori is opened up against a racing opponent**, not only on a fully
  developed board, matching the archetype's multi-Munkidori relocation/heal engine.
  Measured neutral vs the tested Alakazam agents; retained for wider-field value.
- Added `opp_is_racing()` helper (large held hand, or a visible Abra/Kadabra/
  Alakazam draw line).

### Rejected after measurement

- Gating Froslass development and adding an explicit "heal the threatened attacker"
  bias were tested and **lowered** the win rate vs the hand-scaling opponent
  (34.7% -> 33.5%), so both were dropped.  The Froslass-in-losses correlation was
  confounded by game length.

### Results

- vs `alakazam_ml_v11`: 31.8% -> **34.7%** (+2.9pt, z=2.8, 4000 games each).
- vs `alakazam_ml_v10`: 38.6% -> **43.0%** (+4.4pt, z=2.8, 2000 games each).

### Packaging

- Deck unchanged (60 cards, identical SHA to v1/v2).
- Runtime remains self-contained: `main.py`, `policy_base.py`, `deck.csv`.
- 14 static golden-state tests pass.

---

## v2.0.0-final

Data source expanded from 161 to 908 upper-ladder replays by adding 747 analyzable
games from submission 54797424.

### Correctness

- Added Crustle's prevention of attack damage from Pokémon ex.
- Shadow Bullet no longer treats an Active Crustle as a 180-damage KO.
- Boss evaluation now remains available when Crustle is Active.

### Route planning

- Added direct Impidimp + Rare Candy + Grimmsnarl route recognition.
- Removed redundant Morgrem search when the Candy route is complete.
- Punk Up now values the evolution route visible in hand instead of applying a permanent Morgrem bonus.
- Munkidori can receive the manual Darkness Energy earlier when a real Grimmsnarl route exists.

### Tempo

- A live Shadow Bullet is reserved over Froslass development and third-line evolution.
- Completing the second Grimmsnarl remains above the attack.
- Weak chip attacks were not promoted; the larger log shows they are more common in losses.

### Targeting

- Added Dunsparce, Dudunsparce, Dwebble, Crustle, Drakloak, Dragapult ex, mirror evolution bodies, and Mega Kangaskhan ex to route-role evaluation.
- Added survival priority when Adrena-Brain can save a low-HP powered Munkidori.

### Packaging

- Deck remains unchanged.
- Runtime remains self-contained with `main.py`, `policy_base.py`, and `deck.csv`.
