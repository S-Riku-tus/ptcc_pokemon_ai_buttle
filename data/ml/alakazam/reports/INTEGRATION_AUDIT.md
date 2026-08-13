# Original ml_alakazam integration audit

## Scope

This build starts from the user-supplied original `ml_alakazam.zip`. It is not a parallel replacement project. The original agent directory, deterministic v12 fallback, CLI modules, model distillation/export approach, and report layout were retained and upgraded in place.

## High-impact fixes

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
