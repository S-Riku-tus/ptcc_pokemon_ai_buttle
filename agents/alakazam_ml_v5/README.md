# alakazam_ml_v4_candidate — logic-audited build 0.3.0

This build keeps the user-requested 60-card revision and the existing distilled model, while correcting deterministic fallback and ML-boundary problems found during a second full logic audit.

## Deck

- Dunsparce 3 / Dudunsparce 3
- Psyduck 0
- Basic Psychic Energy 3
- Fezandipiti ex 1
- Genesect 1 + Lucky Helmet 1
- Maximum Rod is the only ACE SPEC
- Total 60

## Fezandipiti ex role

Fezandipiti ex is persistent Bench draw support, not an attacker.

- Bench it early when naturally drawn.
- Keep the last Bench slots needed for the first Abra line, the Dunsparce engine, and a real backup attacker.
- Do not fetch it proactively with Telepath Energy.
- Do not attach discretionary Energy or voluntarily promote it.
- Use Flip the Script only after an opposing-turn KO and only when the draw is still useful and deck-safe.
- Delay the optional Bench play when losing one hand card would erase a current KO or worsen the practical hit-count clock.

## Other deterministic corrections

- Evolution hand deltas now include Kadabra/Alakazam draw effects and Rare Candy's full resolution.
- Search cards bypass deckout safety only when that exact search creates an ETA <= 1 backup.
- Search deck costs and net hand changes are card-specific.
- Dawn/Hilda contribute to reachable damage only when their search has a concrete legal goal.
- Survival Bench priority is Abra, Dunsparce, Fezandipiti ex, then Genesect.
- Genesect requires Lucky Helmet already in hand and may not consume required core Bench slots.
- Nighttime Mine is used only when its tax immediately stops the opposing Active Tera attack.

## ML boundary

The model file and confidence thresholds are unchanged. ML is now intent-preserving:

- If fallback selects Abra, ML cannot replace it with Dunsparce.
- If fallback selects a particular evolution stage, ML cannot replace it with another evolution card.
- If fallback selects an attack, ML cannot spend the turn on development.
- Role Pokémon, abilities, trainers, energy, disruption, retreat, and END remain rule-only.

## Validation

See `VALIDATION_REPORT.md` and `LOGIC_AUDIT_V4_1.md`.
