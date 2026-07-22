# v6 Validation Report

Champion = `marnies_grimmsnarl_ex_v5` (rating 723.3, 17-15 over 32 games).
Challenger = `marnies_grimmsnarl_ex_v6` (this agent). Deck is byte-identical, so the
comparison isolates policy quality.

## Static checks

- **Deck**: 60 cards, unchanged from v4/v5 (`test_deck_is_60_and_unchanged`).
- **Self-contained runtime**: `main.py` + `policy_base.py` + `deck.csv`; imports only
  from `cg.api` and the bundled `policy_base`.
- **Golden-state tests**: 48 pass (22 inherited v4 + 10 inherited v5 + 16 new v6).
  - One v4 test was updated for the new opening (Impidimp > Munkidori > Snorunt,
    dropping the "preserve the only Impidimp" heuristic); one v4 test was made
    encoding-robust (`read_text(encoding="utf-8")`).
  - The 16 v6 tests run against the **real** vendor card database, so they exercise the
    true Ability text, ex/mega flags, weaknesses and evolution chains.

## Replay audit (31 recorded v5 episodes, 1963 grimmsnarl decisions)

Every game state where our agent actually acted was replayed through the v6 policy.

| Metric | Result |
| --- | --- |
| Decisions replayed | 1963 |
| `policy_ok` | 1963 |
| `policy_fallback` / `obs_fallback` / exceptions | 0 / 0 / 0 |
| Decisions identical to v5 | 1928 (98.2%) |
| Decisions diverged | 35 (1.8%) |

**Every divergence is an intended fix.** By category (v5 → v6):

- `ATTACK(Shadow Bullet) → END` ×11, `→ PLAY` ×9, `→ ATTACH` ×1 — stop hammering a wall.
- `PLAY → ATTACK` ×6, `ABILITY → ATTACK` ×1 — take a *live* attack v5 was delaying.
- `EVOLVE → PLAY` ×4, `PLAY → EVOLVE/ABILITY/END` ×3 — ETA / opening / target re-ordering.

### Wall behavior

- On the 12 states where v5 fired a 0-damage Shadow Bullet into Crustle, v6 now plays
  **7 END, 3 PLAY, 1 EVOLVE, 1 ATTACH — zero wall attacks.**
- Across all games, v6's 113 Shadow Bullet decisions are **111 live (real damage) + 2
  Bench-KO** (a 40-HP Dwebble finished by Bench-30 + Adrena-Brain). **0 useless
  0-damage wall attacks remain** (v5 had 15 into Crustle + 2 into Ogerpon).
- The Ogerpon game (ep 87374336): v6 marks the Active immune in **every** Ogerpon
  state and develops / ends instead of hitting it for 0 (v5 did not detect it at all).

## What this report does NOT establish

Live win rate. The audit shows v6 is self-contained, crash-free, a strict behavioral
superset of v5's good decisions, and that it removes the specific errors the log
exposed — but the 32-game v5 sample has a wide confidence interval. Run v5 (Champion)
vs v6 (Challenger) on the competition engine, same seeds and swapped seats, ≥ 200
games, and gate promotion on:

| Metric | Target |
| --- | --- |
| Useless 0-damage attacks | 0 |
| First attack by T3 | ≥ 85% |
| First attack on/after T5 | 0 |
| Post-first-attack stall (losses) | ≤ 0.4/game |
| Board wipes | ≤ 5% |
| Low-prize Boss misdirects | 0 |
| crash / illegal action | 0 |
