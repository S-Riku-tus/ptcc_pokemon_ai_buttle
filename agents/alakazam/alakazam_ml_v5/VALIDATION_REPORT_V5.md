# Alakazam ML v5 validation report

Date: 2026-07-18

## Root cause and logic repair

Diagnostics exposed an internal `RecursionError` cycle:

`_turns_to_win -> _ko_active_reachable -> _achievable_hand -> _fez_draw_needed -> _turns_to_win`

`fallback_v12.py` now uses a conservative non-recursive prize clock inside
Fezandipiti draw gating. The active runtime embeds the stable policy in
`fallback_v3.py` while retaining the v5 deck.

## Replay ingestion and training

All five `*_full.zip` bundles under `data/runs/20260717_kaggle_top20` were
ingested.  The loader now reads nested `submission.json` and `episodes.json`
metadata to recover the exact expert seat and deduplicates trajectory IDs.

- 5 teams / 5 submissions / 2 deck clusters
- 3,158 full replay files
- 3,161 usable expert trajectories
- 168,239 aligned ACTIVE/MAIN decisions
- 1,923,948 legal candidate rows
- 5 unresolved decisions
- 99.997% alignment rate

Time-holdout imitation metrics were top-1 59.13%, top-3 84.71%, MRR 73.14%,
and high-confidence accepted top-1 76.58%.  Team and deck holdouts were weaker,
which correctly warned that imitation accuracy alone was not sufficient for
live promotion.


## Reproduction
