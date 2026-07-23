# Marnie's Grimmsnarl ex v7

Rule-based Kaggle agent. v7 keeps every v6 success and repairs the three things
v6 over-corrected while making itself "safer" (audit of the 52-game / 737.3-rating
v6 ladder log, 26-26). The 60-card list is **unchanged** from v4/v5/v6 so v6→v7
isolates policy quality.

**Kept fixed from v6 (do not regress):** complete wall detection (Ogerpon via the
Ability clause), the below-END 0-damage wall attack, the Shaymin/Rabsca bench
shields, and the meta target ranking (anti-Grimmsnarl tech → main Pokémon →
pre-evolutions → other).

**v7 repairs:**

1. **Boss's Orders by purpose.** v6's single `>= 10_000` gate suppressed every
   engine/tempo gust, so Boss collapsed to 0.04/game (v5 was 0.38). v7 scores each
   gust by purpose — **WIN_NOW, WALL_UNLOCK, HIGHER_PRIZE_KO, ENGINE_KO,
   TEMPO_GUST** — each with its own bar, and drops the flat gate. A confirmed
   high-prize Active KO route is still protected from a cheap gust.
2. **Honest, legal-step ETAs.** `first_attacker_eta` / `backup_attacker_eta` now
   respect `appearThisTurn`, require the evolution piece in hand, and (for the
   first attacker) a real path to the Active spot via retreat. A lone Morgrem with
   two Energy and no Grimmsnarl ex in hand is no longer a fake ETA-0 backup.
3. **Fast-race gear vs Mega Lucario / Archaludon** (every attack-less v6 loss was
   vs Mega Lucario): flood Impidimp early, hold optional engines until the first
   attacker is live, and demand a completed backup.
4. **Conditional initial Active.** A sole escape-less Munkidori now ranks below a
   Snorunt that has its Froslass route.
5. **Remembered Dodge/Hide immunity** (best-effort, from attack text).

## Submission payload

- `main.py`
- `policy_base.py`
- `deck.csv`

The full package also contains strategy, changelog, metadata, tests, and
validation notes.
