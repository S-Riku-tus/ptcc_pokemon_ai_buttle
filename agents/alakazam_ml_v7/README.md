# alakazam_ml_v7

v7 combines the v6 deterministic safety fixes and deck with the top21-40
teacher corpus and a newly trained LightGBM ranker.

## Runtime policy

- `fallback_v3.py` remains authoritative.
- The default runtime is shadow-only and always returns the fallback action.
- A future controlled override experiment is limited to same-role Abra or
  Dunsparce bench choices.
- Attack and evolution returned to deterministic fallback because their Top-1
  accuracy declined on the new-teacher future holdout.
- Boss, Ability, Trainer, Energy, Retreat, and the other strategic actions
  remain rule-only.
- `ALAKAZAM_ML_V7_ENABLE_OVERRIDE=1` is for controlled evaluation only.

## Training corpus

The v7 ranker uses the v6 corpus plus ten top21-40 Alakazam rank archives.
The combined rank archives contain sixteen distinct submissions; the ML input
reader resolves metadata per submission sub-bundle.

- 17 ZIPs / 23 submissions / 16 teams / 5 deck clusters
- 6,499 full replay files / 6,518 expert trajectories
- 320,519 aligned decisions
- 3,698,955 legal candidate rows
- 6 unresolved decisions; alignment rate 99.998%

The top21-40 addition contributed 3,249 episodes, 3,265 trajectories, 148,845
decisions, and 1,726,889 candidate rows.

On 34,336 future decisions from the added submissions, Top-1 improved by 0.45
percentage points. Bench improved by 2.58 points, while attack declined by 1.27
and evolution by 0.89. A fixed old-teacher submission holdout also improved for
bench and regressed for evolution, so live scope was narrowed to bench only.

## Deck

v7 retains the v6 list and its last-body Dudunsparce and value-gated Boss
invariants. A rank-2 teacher list ablation replaced one Dudunsparce and Max Rod
with Enriching Energy and Shaymin. It lost the 200-game seat-swapped A/B against
v6, 89-111 (44.5%), with no safety errors, so that deck change was rejected.

## Live evaluation

- rank-2 teacher deck vs v6: 89-111, rejected; v6 deck restored
- bench-only override vs v6 shadow: 102-98, 51.0%, rejected
- default shadow vs non-ML v3: 124-76, 62.0%
- all three 200-game runs had zero crashes, illegal actions, and timeouts

The repository's formal gate also rejected the v3 result because attack-turn
rate was 43.9% versus the configured 70% minimum. v7 therefore remains a
development shadow agent and does not replace v6.

See `data/ml/alakazam_ml_v7_candidate/TRAINING_REPORT.md` and
`reports/top21_40_teacher_analysis.json` under that directory for the complete
audit and replay-level evidence.
