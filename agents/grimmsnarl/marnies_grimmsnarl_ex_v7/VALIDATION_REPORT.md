# v7 Validation Report

Champion = `marnies_grimmsnarl_ex_v6` (rating 737.3, 26-26 over 52 games).
Challenger = `marnies_grimmsnarl_ex_v7` (this agent). Deck is byte-identical, so
the comparison isolates policy quality.

## Static checks

- **Deck**: 60 cards, unchanged from v4/v5/v6 (`test_deck_is_60_and_unchanged`).
- **Self-contained runtime**: `main.py` + `policy_base.py` + `deck.csv`; imports
  only from `cg.api` and the bundled `policy_base`.
- **Golden-state tests**: 61 pass, all against the **real** vendor card database
  (22 v4 + 10 v5 + 16 v6 + 13 new v7).
  - Two v6 initial-Active tests were updated for the new conditional order (a sole
    escape-less Munkidori now ranks below a Snorunt with its Froslass route).
  - The 13 v7 tests cover: purpose-split Boss (WIN_NOW / ENGINE_KO / TEMPO_GUST +
    the preserved Makuhita-over-Mega-Lucario protection), legal-step first/backup
    ETAs (appearThisTurn, piece-in-hand, Active-path), fast_race detection and its
    optional-setup hold, the conditional Active, and temporary Dodge/Hide immunity.

## Live self-play audit (cg engine, 190 games)

| Run | Result | crashes | policy_fallback | illegal | obs_fallback |
| --- | --- | --- | --- | --- | --- |
| v7 vs v6, 150 games | 72-78 (48.0%) | 0 | 0 | 0 | 0 |
| v7 vs {v6, alakazam}, 40 | — | 0 | 0 | 0 | 0 |

- **Runtime safety**: 0 crashes, 0 policy fallbacks, 0 illegal actions, 0 obs
  fallbacks across all 190 games.
- **Boss usage restored**: ~0.28 Boss plays/game (modes observed WIN_NOW /
  HIGHER_PRIZE_KO / ENGINE_KO), up from the log's v6 rate of 0.04/game — the
  single biggest v6→v7 behavioral change, exactly the P0 target.
- **fast_race is correct**: 0 fast_race decisions vs non-fast opponents (the
  earlier Snorunt/Mega-Glalie name-collision false positive is fixed).

## What this report does NOT establish

**Live win rate against v7's target match-ups.** The mirror win-rate (~50%) does
**not** measure v7's gains: a grimmsnarl-vs-grimmsnarl game contains no walls, no
Mega Lucario, and rarely the board states that Boss-by-purpose, honest ETAs and
the fast-race gear were built for. (v5→v6's win-rate delta was likewise not
significant; the behavioral metrics are what moved.)

The right validation, per the v6 audit's own recommendation, is on the competition
engine with v6 (Champion) vs v7 (Challenger), same seeds / swapped seats, ≥ 200
games, **plus ~50 fixed games each vs the hard match-ups** (Mega Lucario,
Jurdon/Archaludon, Dragapult, Fooding/Alakazam, wall decks). Gate promotion on:

| Metric | Target |
| --- | --- |
| Useless 0-damage attacks | 0 (maintain) |
| First attack by T3 | ≥ 85% |
| Games with no attack | < 2% |
| Post-first-attack idle (losses) | ≤ 0.6/game |
| Mega Lucario games with no attack | 0 |
| Reasonable Boss opportunities missed | 0 |
| crash / illegal action | 0 |
| Direct win rate vs v6 | ≥ 53% |

New DIAG counters (`boss_plays`, `boss_modes`, `fast_race_decisions`,
`temp_immunity_hits`) make the next audit measure these directly.
