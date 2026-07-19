# alakazam_ml_v9

v9 is the rule-authority follow-up to the v6-v8 and top-team replay comparison.
The v8 deck and ranker are intentionally fixed: the default runtime still uses
the deterministic policy for strategic choices, and the trained ranker remains
shadow-only unless an experiment explicitly sets
`ALAKAZAM_ML_V9_ENABLE_OVERRIDE=1`.

## Main changes

- Enhanced Hammer now resolves attached-energy options correctly, targets Mist
  before ordinary Special Energy, and reserves the final Hammer in high/medium
  Mist signatures unless removing another Energy creates immediate tempo.
- Fezandipiti ex has explicit `DO_NOT_BENCH`, `DRAW_ONLY`, `PIVOT`, and
  `ALTERNATE_ATTACKER` modes. It can pay one Energy to escape to a ready
  Alakazam, and Cruel Arrow is developed only for Spidops/protection or a
  concrete 100-damage prize route.
- Dual Abra evolution defaults to the Bench but dynamically chooses the Active
  for the only immediate attack, KO, or anti-stranding line.
- Two-hit Boss routes must now be sticky, prize-closing, explicit high-value
  targets. Same-turn KO routes are unchanged.
- Diagnostics expose Hammer targets/reservations, Fezandipiti modes and stalls,
  dual-target Kadabra choices, and same-turn versus two-hit Boss plays.

See `CHANGELOG_V9.md`, `VALIDATION_REPORT_V9.md`, and
`data/runs/ml_v8_evaluation/ML_v6_v8_top_comparison_report.md` for the evidence.
