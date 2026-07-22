# Changelog

## v6

Grounded in the 31-episode v5 ladder log (rating 723.3, 17-15) and the pilot's two
requested fixes. The 60-card list is unchanged, so v5→v6 isolates policy quality.

### Fix 1 — complete wall detection (Ogerpon)

- `_compute_active_wall_blockers()` now matches walls that prevent all damage from a
  *Pokémon with an Ability* (Cornerstone Mask Ogerpon ex) in addition to the *ex*
  clause (Crustle / Sylveon), because Grimmsnarl ex has the Punk Up Ability. Clauses
  that do not apply are still excluded: Basic-only (Farigiraf ex — we are Stage 2),
  Tera-only (Milotic ex), Special-Energy-only (Carracosta — we run basic Darkness).
  Ogerpon (117) is also hard-coded so the guard holds without skill text.
- `bench_damage_lands()` now also treats a benched wall (and Neutralization Zone vs
  non-Rule-Box) as immune to the Bench-30 — the same Ability prevents that 30 too.

  *Log evidence:* the one Ogerpon game (ep 87374336) hit the Active for 0 with no
  detection and was lost with a first attack on turn 28.

### Fix 2 — stop hammering a wall

- A 0-damage Shadow Bullet is scored by `walled_shadow_value()`: worth ending the
  turn on only when the Bench-30 (± a ready Adrena-Brain) takes a prize this turn;
  otherwise it returns -1 and ranks **below END**. Development and the Boss unlock are
  already un-suppressed, so we keep building / gust a real target instead of stalling.
- Morgrem's Corkscrew Punch (60, not blocked by ex/ability walls) is documented as
  the real-damage alternative into a wall.

  *Log evidence:* 15 Shadow-Bullet-into-Crustle 0s across the run (one game hit the
  wall 6 times while holding Boss's Orders on 5 of them), plus 2 into Ogerpon.

### Fix 3 — two-turn Boss's Orders value

- `active_continuation_value()` protects the current Active route: an immediate KO,
  or a 2-3 prize ex/mega that a two-attack Shadow Bullet already KOs. `best_boss_value()`
  now requires a Bench gust to beat that continuation by a margin, so Boss no longer
  trades a high-prize route for a 1-prize chip target — but it still unlocks walls and
  removes KO-able engines.

  *Log evidence:* Boss'd a 1-prize Makuhita while a 3-prize Mega Lucario ex Active was
  a two-attack KO; similar low-prize gusts across the losses.

### Fix 4 — meta target priority

- `target_priority_bonus()` replaces v5's 3-tier `route_piece_bonus` with the pilot's
  explicit ranking: anti-Grimmsnarl tech (Shaymin/Rabsca, walls) → main Pokémon
  (ex/mega by prizes, then draw/damage engines) → pre-evolutions → other. Pre-evolutions
  are detected dynamically from the card database's `evolvesFrom` chain (any Basic that
  leads to a Mega ex, e.g. Riolu, is a real snipe) and boosted when the evolved form is
  already on the opponent's board. `route_piece_bonus` remains as an alias.

### Fix 5 — faster, safer opening

- `first_attacker_eta()` / `backup_attacker_eta()` express the route as whole turns;
  `backup_is_close()` now means `backup_attacker_eta ≤ 1`.
- `opening_focus()` (no attacker yet, `first_attacker_eta ≥ 2`) boosts line-finding
  searches and suppresses optional setup (second Munkidori, Froslass, Handheld Fan).
- Initial Active preference changed to Impidimp > Munkidori > Snorunt (ladder data),
  dropping v4/v5's "preserve the only Impidimp" heuristic.
- A lone board (`board_count ≤ 1`) force-develops a Basic above everything so a single
  KO cannot wipe us.

### Tests & validation

- 32 inherited v4/v5 golden-state tests still pass (one v4 test updated for the new
  Active-preference; one made encoding-robust). 16 new v6 tests run against the real
  vendor card database and cover Ogerpon detection, the below-END wall attack, the
  two-turn Boss value, the target tiers, the bench shields, and the ETA/opening.
- Replay audit over all 31 episodes (1963 grimmsnarl decisions): 0 exceptions, 0 policy
  fallbacks; 98.2% of decisions identical to v5; every divergence is an intended fix;
  0 useless wall attacks remain (v6's 113 Shadow Bullets are 111 live + 2 Bench-KO).

## v5

- Generalised the ex-damage-prevention wall beyond Crustle (Crustle/Sylveon +
  Neutralization Zone), added `live_attack_ready()` so walled turns do not suppress
  development, and valued a Boss's Orders wall unlock.
- Added evolve-on-the-Bench rules (Morgrem only into the Active when it can attack
  this turn; Grimmsnarl onto the Bench rather than into a wall).

## v4

- Route-aware pre-attack sequencing, useful second Munkidori before Shadow Bullet,
  Rare Candy that completes the backup above the live attack, unified Poké Pad /
  Spikemuth search, reworked Unfair Stamp, one-turn prize-route target value.
