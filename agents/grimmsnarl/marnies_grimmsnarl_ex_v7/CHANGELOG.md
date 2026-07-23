# Changelog

## v7

Grounded in the 52-game v6 ladder audit (rating 737.3, 26-26). v6 fixed v5's
biggest wall bugs — 0 useless 0-damage attacks and a faster opening — but making
the wall/opening logic *safer* over-corrected Boss usage, backup evaluation and
early engine development. v7 keeps every v6 success fixed and repairs the
over-corrections. The 60-card list is unchanged, so v6→v7 isolates policy quality.

### Fix 1 — Boss's Orders by purpose (P0, biggest regression)

- v6 routed every Boss through one flat `best_boss_value() >= 10_000` gate, but
  the "gust a key engine" branch topped out around 8_500 and never cleared it, so
  Boss almost vanished (**0.04/game vs v5's 0.38**).
- `best_boss_value()` is rebuilt to score each gust by *purpose*, each with its
  own bar, and the flat gate is dropped (`reserve_adjust`, the Petrel search and
  `score_play_trainer` now gate on `> 0`):
  - **WIN_NOW** — the gusted KO takes our remaining prize(s); always top.
  - **WALL_UNLOCK** — Active is a wall; move a hittable body into range.
  - **HIGHER_PRIZE_KO** — KO a benched body worth MORE prizes than the Active.
  - **ENGINE_KO** — KO a key engine / tech / sole attacker at ≥ equal prizes,
    unless that throws away a *confirmed* ≥-prize Active KO this turn.
  - **TEMPO_GUST** — strand a hard-to-retreat body we cannot KO to buy a turn,
    only when the Active route is not itself a KO this turn.
- A confirmed / near-confirmed high-prize Active KO route is still protected from
  a cheap gust — the Makuhita-over-Mega-Lucario mistake stays fixed — but the old
  fixed +6_000 pad is gone, so engine/tempo gusts finally fire.

  *Log evidence:* 52 games, only 2 Boss plays, while many turns held Boss with a
  live attacker and a gustable engine/high-prize target.

### Fix 2 — honest, legal-step attacker ETAs (P0)

- `first_attacker_eta` / `backup_attacker_eta` are rebuilt on legal steps:
  - an evolve needs the piece in hand **and** a body that did not appear/evolve
    this turn (`appearThisTurn`);
  - the *first* attacker also needs a real path to the Active spot — retreat is
    the only switch this deck runs (`_active_can_retreat`, Handheld-Fan aware);
  - the backup does not need the Active this turn (it promotes on a KO), but it
    must be a real route — a lone Morgrem with two Energy and **no Grimmsnarl ex
    in hand is no longer ETA 0** (v6's exact over-count).

  *Log evidence:* post-first-attack idle nearly doubled (0.44→0.83) because v6
  treated un-completable "backups" as ready and stopped building the real one.

### Fix 3 — fast_race gear vs Mega Lucario / Archaludon (P0)

- Every attack-less v6 loss (3 games) was vs Mega Lucario. `fast_race()` (Mega ex
  on board, or a Riolu/Lucario/Duraludon/Archaludon line) switches gears:
  flood Impidimp early (a KO cannot wipe the line), hold optional engines (2nd
  Munkidori / Froslass) until the first attacker is live (`hold_optional_setup`),
  and require a *completed* (ETA 0) backup, not merely ETA 1.
- Detection is name/megaEx based on purpose; it deliberately does **not** use the
  DB's name-unioned prize-potential map, which collides (Snorunt inherits a Mega
  Glalie line) and would falsely flag the mirror.

### Fix 4 — conditional initial Active (P0)

- Impidimp is still best. A *sole* Munkidori with no Darkness to pay its retreat
  now ranks **below** a Snorunt that has its Froslass route, so we do not strand
  our only Adrena-Brain engine in the Active (v6's Munkidori start won only 35%).

### Fix 5 — remembered temporary immunity (P0, best-effort)

- A coin Dodge/Hide attack that prevents all damage during our next turn now
  zeroes that Pokémon (`temp_immune`, keyed by serial) for the turn in
  `shadow_damage`, `active_target_immune_to_ex` and `bench_damage_lands`, so we do
  not waste a Shadow Bullet / Boss into it. The attack set is derived from card
  text (dropping Basic-only / damage-threshold clauses we pierce); log parsing is
  fully defensive, so an unknown log schema simply leaves it inert.

### Unchanged (v6 successes kept fixed)

- Wall detection (Ogerpon via the Ability clause), the below-END 0-damage wall
  attack, the Shaymin/Rabsca bench shields, and the meta target priority.

### Diagnostics (P2)

- New DIAG: `boss_plays`, `boss_modes`, `fast_race_decisions`, `temp_immunity_hits`.

### Tests & validation

- 61 static golden-state tests pass (22 v4 + 10 v5 + 16 v6 + 13 new v7 against the
  real vendor DB). Two v6 initial-Active tests were updated for the new
  conditional order.
- 190 real self-play games (150 v7-vs-v6 + 40 mixed) via the cg engine: 0 crashes,
  0 policy fallbacks, 0 illegal actions, 0 obs fallbacks. Boss usage restored to
  ~0.28/game; fast_race no longer false-fires on non-fast decks. The mirror
  win-rate (~50%) does not measure v7's match-up-specific gains — see
  VALIDATION_REPORT.md.

## v6

Ladder-grounded fixes over v5: complete wall detection (Ogerpon), stop hammering
walls, two-turn Boss's Orders, meta-aware bench/Adrena/Boss targeting, and a
faster/safer T3 opening. (Full detail retained from the v6 changelog history.)

## v5

Generalised the ex-damage wall, added `live_attack_ready()`, valued a Boss wall
unlock, and added evolve-on-the-Bench rules.

## v4

Route-aware pre-attack sequencing, useful second Munkidori, Rare Candy that
completes the backup, unified search, one-turn prize-route target value.
