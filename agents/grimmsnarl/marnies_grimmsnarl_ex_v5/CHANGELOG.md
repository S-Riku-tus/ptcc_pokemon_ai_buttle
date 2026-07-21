# Changelog

## v5

Two Pokémon-TCG fundamentals, grounded in the 43-episode v4 ladder log
(rating 658.1, 60% win rate in-sample) and the Alakazam line's "locked" /
"evolve where it is safe" principles. The 60-card list is unchanged.

### Fix 1 — stop hammering a damage-immune wall

- Generalised the ex-damage-prevention detection beyond hard-coded Crustle:
  Crustle **and** Sylveon (any-ex blockers) plus a Neutralization Zone stadium
  against non-Rule-Box Pokémon. Farigiraf ex is deliberately excluded (it only
  stops *Basic* ex attackers; Grimmsnarl ex is Stage 2).
- Added `live_attack_ready()` — Shadow Bullet is only a *reserved attack* when it
  does non-zero damage. Against a wall the attack is worthless, so pre-attack
  development, backup building and searches are **no longer suppressed**.
- `best_boss_value()` now values a Boss's Orders that gusts *any* reachable Bench
  target into the Active as an unlock — even without an immediate KO — because
  180 to a fresh target beats 0 to the wall.
- The 0-damage Shadow Bullet dropped from 650k to `520k + Bench-30 value`, below
  all development and the Boss unlock.

  *Log evidence:* 24 Shadow-Bullet-into-Crustle actions across 6 episodes in v4
  (one game hit the wall 12 times); 17 left a gustable, KO-able Bench target unused.

### Fix 2 — evolve on the Bench, not into a grave

- Morgrem has no Punk Up and needs two Energy already attached to attack, so v5
  only evolves it in the Active when it can attack *this turn*; otherwise it
  prefers a Bench body (and de-prioritises an exposed Active Morgrem when the
  opponent's Active can KO it).
- Grimmsnarl ex keeps its Active preference (Punk Up fuels it to attack the turn
  it evolves) **except** into a wall it cannot damage, where the Bench copy is
  preferred so Boss's Orders can later open a real target.
- Added `opp_active_max_damage()` / `opp_active_threatens()` /
  `bench_evolve_available()` helpers.

  *Log evidence:* 10 Active evolutions in v4 where a Bench evolve of the same
  card was legal; 9/10 failed to attack that turn, 10/10 hit a failure condition.

### Tests

- 22 inherited v4 golden-state tests still pass (no regression).
- 12 new v5 tests cover the generalised immunity, the locked-state Boss unlock,
  the un-suppressed development, and the Bench-vs-Active evolution rules.

## v4

- Kept the upper-ladder 60-card list unchanged.
- Added route-aware pre-attack sequencing.
- Allowed an immediately useful second Munkidori before Shadow Bullet while reserving bench slots for the Marnie line.
- Made Rare Candy that completes the second attacker a BUILD_BACKUP action above the live attack.
- Unified Poké Pad and Spikemuth Gym with the direct Candy-route model; removed redundant Morgrem search.
- Added missing-entire-line detection to non-rule Pokémon search.
- Reworked Unfair Stamp around both hand sizes and protection of a unique backup route.
- Added one-turn prize-route value to Shadow Bullet and Adrena-Brain target selection.
- Added eight v4-specific golden-state tests; 22 total tests pass.
