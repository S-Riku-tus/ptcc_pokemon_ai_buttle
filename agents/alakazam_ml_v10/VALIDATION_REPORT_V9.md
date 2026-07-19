# v9 validation report

## Scope

v9 is a rules-only challenger built from v8. The deck and
`ranker_model.json` are unchanged so that the replay-derived Hammer,
Fezandipiti ex, evolution, and Boss changes can be evaluated without mixing in
a deck or model change. The model remains shadow-only by default.

## Static and unit validation

- `scripts/validate_agent.py --agent alakazam_ml_v9`: passed; 60-card deck,
  no warnings.
- v9 source/runtime tests: 40 passed.
- Model SHA-256:
  `e9a47de0bf27e9a5528eb09b846f9e63a530e1e8ab6c142eaf62faba5a3cdba3`
  (identical to v8).

## Real replay regression

The v9 policy was replayed against the actual Enhanced Hammer target
observations from the three v8 Mist episodes `86806338`, `86800296`, and
`86798142`.

| Observation class | Selected / offered | Result |
| --- | ---: | --- |
| Mist Energy was offered | 5 / 5 | Mist selected |
| Mist Energy was absent | 4 / 4 | best available non-Mist target selected |

This confirms both v9 context fixes that were missing from v8: effect cards are
read from the current `select.effect` schema, and attached Energy choices are
resolved through `energyIndex` instead of scoring their owner Pokemon.

## Local arena smoke test

Command:

```text
python scripts/local_arena.py agents/alakazam_ml_v9 agents/alakazam_ml_v8 --games 100 --seed 741
```

Result: v9 48, v8 52, draws 0. Both agents had zero crashes, zero illegal
selections, and zero policy/observation fallbacks. This is a stability smoke
test, not evidence of a rating improvement: the local mirror does not reproduce
the specific Mist, Spidops/protection, and Fezandipiti stall distributions that
motivated v9.

## Repository-wide test status

The full repository test command completed with 15 failures unrelated to the
new v9 directory. They are attributable to pre-existing missing agent fixtures
(`alakazam_ml_v2_expanded` and `kashiwashira_spidops_reconstruction_v1`) plus an
existing `alakazam741_v10` Battle Cage route assertion. The isolated v9 suite
passes, so these failures were recorded rather than repaired outside this
change's scope.

## Promotion decision

Keep v9 as a challenger. It fixes the replay-confirmed Hammer targeting fault
and adds explicit controls for the other high-value failure modes, but it
should not replace v8 until fresh ladder or reconstructed multi-matchup games
meet the gates in `CHANGELOG_V9.md`.
