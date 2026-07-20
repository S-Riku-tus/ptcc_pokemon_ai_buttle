# alakazam_ml_v10

v10 is a stability-focused Alakazam challenger built from the v9 ladder audit,
the saved top-40 replay corpus, and the current leaderboard snapshot from
2026-07-19.  It keeps the evidence-backed v9 deck and v9's Hammer, Fezandipiti,
Kadabra-target, and strict Boss authority.

## Runtime changes

- A general one-Energy support pivot connects a stranded Active Shaymin,
  Dunsparce, Abra, Kadabra, or Fezandipiti ex to a powered benched Alakazam in
  the same turn. Dudunsparce is excluded because Run Away Draw is its free
  escape route.
- The ML ranker now distinguishes semantically meaningful alternatives from
  interchangeable hand copies. Selecting another copy of the same Abra is no
  longer counted as a model override.
- Strategic ML remains shadow-only by default. An experiment must explicitly
  set `ALAKAZAM_ML_V10_ENABLE_OVERRIDE=1`, and the current bench allowlist still
  has no promoted meaningful alternative.

## ML changes

- 323,889 aligned decisions / 3,736,551 legal candidates / 274 features.
- Added board-body, backup-route, forced-draw runway, deck-pressure,
  support-pivot, Shaymin/spread, Grimmsnarl, and Froslass interactions.
- Added bounded submission/team/action balancing so one very large submission
  cannot dominate the multi-teacher corpus.
- The new ranker improved unseen-team Top-1 and unseen-submission accepted
  Top-1, but regressed slightly on time and deck holdouts, so it is retained for
  shadow diagnostics rather than live control.

## Deck decision

The submitted deck remains the v9 list with Shaymin and Max Rod. In controlled
500-game, one-card deck ablations, Enriching Energy scored 44.4% against Max
Rod. A third Dudunsparce improved the mirror but was effectively tied against
the Grimmsnarl generic deck; Shaymin's unique bench-damage protection and the
top-replay deck evidence break the tie.

See `CHANGELOG_V10.md`, `VALIDATION_REPORT_V10.md`, and the reports under
`data/ml/alakazam_ml_v10_candidate/reports/`.
