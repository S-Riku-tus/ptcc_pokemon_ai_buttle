# Changelog

This final v2 supersedes the preliminary v2 package and includes all additional fixes derived from submission 54797424.

## v2.0.0-final

Data source expanded from 161 to 908 upper-ladder replays by adding 747 analyzable games from submission 54797424.

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
