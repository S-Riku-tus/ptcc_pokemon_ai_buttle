# Grimmsnarl ML v4.5

`grimmsnarl_ml_v4_5` is an isolation experiment requested after the v5.1 data
refresh. It copies v4 and changes only its ranker model and metadata.

- v4 code, fallback policy, planner, features, deck, and policy base: exact
  SHA256 matches.
- Ranker: the same 2,000-tree, 823-feature model promoted to v5.1.
- Teacher pin: 16494330, unchanged from v4.
- Offline refreshed-test Top-1: 0.8500 vs old v4 model 0.8465.
- New rank-3 team Top-1: 0.8127 vs 0.7850.
- Frozen-v4 test Top-1: 0.8523 vs 0.8503; no detected regression.
- Copied v4 tests: 116/116 passed.
- Local safety smoke vs v4: 14-6 over 20 games, no crashes or illegal
  selections; 34.15 ms/move vs 21.05 ms/move. Twenty games are not treated as
  a precise strength estimate.

The detailed data selection and model sweep are in
`../grimmsnarl_ml_v5/DATA_REFRESH_2026-08-06.md`.
