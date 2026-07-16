# Original ml_alakazam integration audit

## Scope

This build starts from the user-supplied original `ml_alakazam.zip`. It is not a parallel replacement project. The original agent directory, deterministic v12 fallback, CLI modules, model distillation/export approach, and report layout were retained and upgraded in place.

## High-impact fixes

1. `src/replay_io.py` now accepts both `replay/episode_*.json` and `replays/episode_*.json`.
2. Seat inference uses source manifest, exact TeamNames, modal target-deck evidence, restricted aliases, and safe self-play handling. Ambiguous cases are excluded.
3. Labels use the same-seat action stored at replay step t+1 as an index into the legal `select.option` list.
4. Dataset construction supports team, submission, deck, and chronological holdouts.
5. Policy features grew from the original 76 to 225 observation/candidate-only features, including explicit state-action interactions.
6. Rank/deck/outcome/quality weights are mild and bounded because strong weighting failed ablation.
7. LightGBM categorical splits are now distilled correctly as category sets (`lightgbm_tree_v2`). The original numeric-only distiller could not safely export the expanded ranker.
8. The existing v12 fallback remains authoritative for Boss, Retreat, Xerosic, and Hammer. Energy requires probability >= 0.85 and margin >= 0.12.
9. Old battle artifacts are marked stale rather than presented as results of the new model.
10. Pipeline output uses compressed CSV when PyArrow is unavailable; Parquet remains an optional acceleration artifact.

## Actual replay recheck

The integrated parser and manifest builder were rerun against all 20 supplied ZIPs:

- 2,058 full replay files
- 1,894 recovered from plural `replays/` paths
- 2,074 target trajectories
- 19 teams, 20 submissions, 8 deck hashes
- regenerated manifest key columns exactly matched the bundled manifest

## Validation

- `compileall`: passed
- pytest: 16 passed
- singular/plural dataset smoke: 67 decisions, 480 candidates, alignment 1.0
- 95,254-decision stored dataset integrity checks: passed
- native LightGBM vs categorical distilled runtime: maximum absolute error < 1e-12 in test; earlier 500-row audit was 0.0
- replay-observation legal-action smoke: 40 decisions, 0 illegal actions, 0 exceptions
- official battle-engine evaluation: not run because `vendor/cg` and the official opponent harness were not present

## Compatibility note

The directory name `agents/alakazam_ml_v1` is retained so existing scripts and paths do not break. The metadata and exported artifact identify the implementation as `alakazam_ml_v2_expanded`.
