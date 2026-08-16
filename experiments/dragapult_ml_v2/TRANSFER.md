# What transfers from the Grimmsnarl/Alakazam line, and what does not

The Grimmsnarl line ran 29 versions and the Alakazam line 36. Most of that work
is deck-specific, but the parts that are not are worth naming explicitly so the
next deck does not rediscover them.

## Transfers as-is (deck-agnostic machinery)

| Asset | Where it lives now | Why it is portable |
|---|---|---|
| Generated card rule tables | `scripts/build_dragapult_card_tables.py` | The observation exposes only `id/hp/maxHp/energies/tools`. Prize value, weakness, printed HP and the attack-cost frontier have to be resolved from the card database or the model cannot express "this attack kills". |
| `prize_value` (Mega ex 3 / ex 2 / else 1) | `ml_features.prize_value` | Prizes are the win condition in every deck. |
| `best_printed_damage` / `damage_against` / `incoming_damage` | `ml_features` | Prices "does our Active survive their turn" for any attacker/defender pair. |
| Pilot-conditioned training: `--team-feature --split-mode per-team --episode-equal-weight --teacher-equal-weight` | `train_grimmsnarl_v2_teacher.py` | Beats both pooled and single-teacher training; one long game or one prolific pilot cannot dominate. |
| Strict Top-1 early stopping, patience 800 | same | NDCG-stopping shipped half a model on Alakazam v33. Patience is cheap; for Dragapult it was worth only +0.0011, but it costs one run to check. |
| Intra-turn history columns (offer/passed-over counters) | `ml_runtime.TURN_FEATURES` | Four of the top twelve features by gain in the Dragapult model. |
| Per-own-turn denominators for uptake metrics | `analyze_dragapult_engine_uptake.py` | 18.9% vs 88.6% on the same Grimmsnarl data depending on the denominator. `current.turn` is shared between seats, so own-turn ordinals must be reconstructed. |
| Exact-deck-hash corpora, re-hashed per seat | `collect_exact_deck_teachers.py` | Archetype labels mix incompatible lists. |
| Refetch teachers every iteration | — | The EpisodeService keys episodes by submission id and only serves the latest 1,000. On 2026-08-16, 6 of the 8 exact-list Dragapult teams were new teams and 0 of the 8 submission ids were in the 08-14 selection. |
| "Guards that override imitation lose" | — | Grimmsnarl v24 (−72.7 Elo at 100% binding), the Alakazam safety shell (−5.16 points), and now Dragapult v1.1 (2,322 decisions at 0.402 against the model's 0.727). |
| Bucket ladder results by *opponent* rating, never own | — | Own-rating bucketing is mean reversion; ratings also deflate by calendar day. |

## Transfers as a pattern, rewritten per deck

| Grimmsnarl form | Dragapult form |
|---|---|
| `shadow_damage_to` — Shadow Bullet 180, doubled on Dark-weak, zero into ex-damage walls | `phantom_damage_to` — Phantom Dive 200, **never** doubled (nothing in the pool is Dragon-weak), zero into the same three walls |
| `bench_snipe_lands` — the Bench-30 is attack damage and a wall shrugs it off | Phantom Dive's six counters are *counters*, not damage, so the walls do not apply; only the counter budget does |
| `turn_routes` — prizes each Boss target yields from 180 + Bench-30 | prizes from 200 to the Active plus the best greedy placement of six counters (`spread_prizes`) |
| Punk Up energy allocation on evolving into Grimmsnarl ex | Crispin's two-step (choose an Energy from deck, then the body), which is where the route ETA has to be computed against `contextCard` |
| `movable_counters` / `heals_needed` for Adrena-Brain | **identical card, identical wording** — ported unchanged |
| Morgrem is the wall breaker because Grimmsnarl ex is an ex | Drakloak (70) and Munkidori (60) are the only bodies that can damage Crustle/Sylveon/Cornerstone Ogerpon; Dragapult ex does zero |

## Does not transfer

- Froslass / Freezing Shroud, the shroud ledger and `ABILITY_POKEMON_IDS` as a
  *targeting* set. Dragapult has no ability that punishes ability holders. The
  generated set is still emitted, but only as a board-composition count.
- Unfair Stamp escalation, Petrel, Spikemuth Gym, the dead-Stamp gradient.
- Rare Candy routing: Dragapult has no Rare Candy, so the evolution route is
  strictly two steps and `route_eta` is exact rather than a lower bound.
- Everything about going first/second. Grimmsnarl's whole going-second deficit
  was Grimmsnarl ex uptime; Dragapult's own-turn curve is symmetric so far.
- The Grimmsnarl matchup map. Dragapult's worst common cells are Conkeldurr and
  Mega Kangaskhan (teachers 0.500) and Hydrapple (0.548); its best are Alakazam
  (0.767) and Marnie's Grimmsnarl (0.749).

## New here, worth carrying forward

1. **Typed-cost attackers need per-colour columns.** A total energy count
   cannot express a two-colour attack cost. This cost Dragapult v1 2.4 own
   turns of tempo. Any future deck with a typed cost needs the same three
   columns: per-colour count on the body, ETA after the candidate action, and
   "does this complete the cost".
2. **`OptionType.ABILITY` carries `area`/`index`, not `inPlayArea`/
   `inPlayIndex`.** Reading the wrong pair fails silently. v1 had it wrong in
   both the feature module and the rule policy.
3. **`remainingOverageTime` must never be a feature.** It records the pilot's
   compute cost, and our distribution barely overlaps the teachers'.
4. **Multi-pick contexts can be imitated without a ranker** by measuring the
   teachers' empirical take rate per card on the training split and using it as
   the deterministic order. Ultra Ball's discard went from 0.210 to 0.448
   agreement this way, with no model involved.
