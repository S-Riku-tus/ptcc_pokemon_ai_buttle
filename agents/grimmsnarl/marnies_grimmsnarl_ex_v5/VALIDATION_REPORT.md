# Validation Report — Marnie’s Grimmsnarl ex v5

## Scope

Static + replay-driven validation. The official competition engine and external
matchup harness were not available, so no claim is made about live win rate.

## Results

- Deck count: **60**
- Deck changed from v4: **No**
- Python compile: **Pass**
- Self-contained imports: **Pass**
- Golden-state tests: **34/34 pass** (22 inherited v4 + 12 new v5)
- Runtime import against the real card DB: **Pass**
  (`EX_ACTIVE_BLOCKERS = {330 Sylveon, 345 Crustle}`, Farigiraf 83 correctly excluded)
- Replay smoke test over 4 Crustle-wall episodes: **388 grimmsnarl decisions,
  0 exceptions, 0 policy fallbacks**
- Submission payload: **main.py, policy_base.py, deck.csv only**

## Log evidence driving v5 (v4 sub 54868864, 43 episodes, rating 658.1)

- **Bug 1 (wall):** 24 Shadow-Bullet-into-Crustle actions across 6 episodes;
  one game hit the wall 12 times; 17 left a gustable, KO-able Bench target unused.
- **Bug 2 (evolve):** 10 Active evolutions where a Bench evolve of the same card
  was legal; 9/10 failed to attack that turn, 10/10 hit a failure condition.

## v5-specific guarantees (v4 → v5 divergence audit on identical observations)

1. Damage-immune detection generalises to Crustle, Sylveon and Neutralization
   Zone (non-Rule-Box); Farigiraf ex (Basic-only) is excluded.
2. Against a wall, `live_attack_ready()` is False, so development/search/backup
   are no longer suppressed (audited: v4 capped everything to ~730k and made a
   marginal play; v5 searches/draws toward the unlock).
3. Boss's Orders is valued as an unlock even without an immediate KO while walled.
4. The 0-damage Shadow Bullet ranks below all development and the Boss unlock
   (650k → ~520k), so the wall is no longer hammered.
5. Morgrem evolves into the Active only when it can attack this turn; otherwise a
   Bench body is preferred (audited: ep 87209198 now evolves Grimmsnarl on the
   Bench vs the Crustle wall instead of the Active).
6. Grimmsnarl ex keeps its Active preference except into a wall it cannot damage.

## Remaining uncertainty

Live engine validation and matchup evaluation must be performed by the user. The
recommended comparison is v4 vs v5 with identical seeds and seat swaps, tracking:
attacks into damage-immune walls (should approach 0 when a Boss/develop line
exists), Boss unlocks played, and evolutions placed on the Bench while the
opponent's Active threatens the freshly-evolved body.
