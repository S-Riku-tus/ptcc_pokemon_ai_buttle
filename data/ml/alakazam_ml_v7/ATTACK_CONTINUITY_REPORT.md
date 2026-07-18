# Alakazam ML v7.1 Attack Continuity Audit

Date: 2026-07-18 JST

## Decision

The low legacy `attack_turn_rate` was primarily a metric-definition problem,
not evidence that v7 declines legal attacks. The metric counted every engine
turn on which the agent made any selection, including setup and non-MAIN
selections. It remains in reports for historical comparison, but it is no
longer a promotion gate.

v7.1 remains shadow-only. Attack, evolution, retreat, and end remain
deterministic fallback actions. The new ranker is useful for research, but no
live override scope was expanded and v6 remains the champion.

## Rank-1 Replay Evidence

Source:
`data/runs/kaggle_top50/latest_rank01_Majkel1337_54662660_top_54662660.zip`

- 164 replays; 163 finished; 129 wins (79.14%)
- legacy all-acting-turn attack rate: 35.35%
- MAIN-turn attack rate: 48.61%
- attack-opportunity conversion: 96.07%
- post-first-attack opportunity conversion: 97.49%
- average first attack: own MAIN turn 2.25
- attacks/game: 5.21
- attackable END choices: 8 total

The teacher had 678 post-first-attack MAIN turns with no attack option. The
Active Pokemon was Fezandipiti ex on 600 of those turns. Therefore a low raw
rate is compatible with the rank-1 wall-and-rebuild strategy and high win
rate. Forcing attacks or removing wall turns would optimize against the
teacher evidence.

Detailed replay rows:
`data/ml/alakazam_ml_v7/reports/rank01_attack_continuity.json`

## Local v7 Evidence

Fresh 200-game v7 default-shadow versus v6:

- v7 107-93 (53.5%; 95% CI 46.6%-60.3%)
- legacy all-turn rate: 41.90%
- MAIN-turn attack rate: 68.33%
- attack-opportunity conversion: 100%
- missed attack-opportunity turns / attackable END: 0 / 0
- attacks/game / Alakazam attacks/game: 5.34 / 3.985
- MAIN idle after first attack in losses: 1.83
- deckout / boardout: 11.0% / 5.5%

Fresh 200-game v7 default-shadow versus non-ML v3:

- v7 111-89 (55.5%; 95% CI 48.6%-62.2%)
- legacy all-turn rate: 42.73%
- MAIN-turn attack rate: 69.61%
- attack-opportunity conversion: 100%
- missed attack-opportunity turns / attackable END: 0 / 0
- attacks/game / Alakazam attacks/game: 5.29 / 3.97
- MAIN idle after first attack in losses: 1.66
- deckout / boardout: 5.0% / 10.0%

Both runs pass the corrected attack-continuity checks. They remain REJECT due
to confidence/resource/board safety checks, not because the policy skips an
offered attack.

## Evaluator Fix

The promotion gate now uses:

- minimum attack-opportunity conversion: 95%
- maximum MAIN idle turns after first attack in losses: 2.0

The legacy all-turn rate is still emitted as a diagnostic. Events and saved
trajectories now record `is_main_decision` and `attack_offered`, so each
denominator is auditable.

## Learning Update

The requested rank-1 ZIP overlaps 120 of its 164 episodes with the existing
v7 corpus. Deduplication added 44 net trajectories rather than weighting the
same teacher twice.

Full v7.1 corpus:

- 18 ZIPs; 23 submissions; 16 teams; 5 deck clusters
- 6,663 replay files; 6,562 usable trajectories
- 322,167 aligned decisions; 3,717,660 candidates
- 237 features; 6 unresolved decisions; 99.9981% alignment
- model SHA256: `8578b74c31263e4f2032836f59bd9b946b8616dea197c62839071e747de6d0ec`

Twelve public-state features distinguish a ready Active Alakazam from a ready
Bench backup and add interactions for END, retreat, Dudunsparce repositioning,
and active-versus-backup evolution.

On the exact same time split, with only those 12 features fixed to zero, the
enabled feature set changed Top-1 as follows:

- overall: +0.84 percentage points
- accepted decisions: +1.25 points
- END: +16.46 points
- retreat: +13.99 points
- bench: +0.05 points
- attack: -0.29 points
- evolve: -0.25 points

This supports the new state representation for continuity decisions, but does
not support granting the model attack or evolution authority. Exact ablation:
`data/ml/alakazam_ml_v7_candidate/reports/attack_continuity_feature_ablation.json`

## Training Safety Fix

During the first retrain attempt, the top21-40 directory had been renamed but
the config still used the old path. The pipeline silently trained an 8-ZIP
partial corpus. That model (`be456d...13feb`) was immediately removed from the
runtime, and the prior `53427a...15ce0` model was restored before retraining.

The config now points to `data/runs/20260718_kaggle_top21_40`. The training
pipeline also fails when any explicit `replay_roots` path is missing, preventing
future partial-corpus runs from being reported as successful.

## Remaining Priorities

1. Reduce boardout without sacrificing the 100% attack-opportunity conversion.
2. Reduce mirror deckout by improving optional Dudunsparce draw timing.
3. Evaluate first-attack latency only after the board/deck safety regressions.
4. Keep attack/evolve rule-only until an exact-split offline gain is followed
   by a 200-game controlled live gate.

Artifacts:

- v6 gate: `data/runs/ml_v7_evaluation/attack_continuity_gate/20260718_173640_alakazam_ml_v6_vs_alakazam_ml_v7`
- v3 gate: `data/runs/ml_v7_evaluation/attack_continuity_v3_gate/20260718_173753_alakazam741_v3_vs_alakazam_ml_v7`
- candidate pipeline: `data/ml/alakazam_ml_v7_candidate/reports/pipeline_report.json`
- diagnostic script: `scripts/analyze_attack_turn_continuity.py`
