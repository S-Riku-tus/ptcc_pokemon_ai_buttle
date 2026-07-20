# v7 changelog

## v7.1 attack-continuity audit

- Added the requested rank-1 Majkel1337 bundle: 164 replays, 44 net new after
  deduplicating 120 overlaps.
- Replaced the impossible 70% all-engine-turn promotion gate with 95% attack
  opportunity conversion and a MAIN-only post-first-attack idle check.
- Added 12 active-versus-backup attacker features and retrained on 322,167
  decisions / 3,717,660 candidates.
- Kept attack and evolution rule-only after the exact feature ablation did not
  improve their Top-1 accuracy; runtime remains shadow-only.
- Made raw-replay training fail when an explicit replay root is missing, after
  detecting and rejecting one partial-corpus retrain caused by a moved folder.

- Fixed ML ingestion for ZIPs containing multiple submission sub-bundles.
- Added 3,249 top21-40 Alakazam episodes from 16 submissions.
- Retrained the 225-feature LightGBM ranker on 320,519 decisions.
- Limited future ML override experiments to guarded bench decisions.
- Returned attack and evolution to deterministic fallback after holdout
  regressions.
- Tested the rank-2 reference deck (Dudunsparce -1, Max Rod -1, Enriching
  Energy +1, Shaymin +1), rejected it after an 89-111 A/B loss, and restored
  the v6 deck.
- Kept the normal runtime shadow-only while running the 200-game override gate.
- Rejected the bench-only live override after a 102-98 result against v6 did
  not meet the 53% win-rate gate.
- Confirmed default-shadow v7 beat non-ML v3 124-76, while the formal report
  still failed the 70% attack-turn-rate requirement.
