# Final report

## 1. Conclusion

The audited ranker reaches semantic Top 1 74.17% and Top 3 90.59%, passing Gate 2. The conservative threshold-0.65 hybrid is submission-compatible at the Python level and had zero illegal actions/crashes locally. Its latest-logic head-to-head improvement is not statistically established, and it remains weaker than v9/v11 in the recorded mirror tests; Gate 3 is therefore partial, not claimed complete.

## 2. Alakazam teams and deck variants

- `Majkel1337` rank 1: `majkel_exact` (164 episodes)
- `Majkel1337` rank 2: `majkel_exact` (100 episodes)
- `Yushin Ito` rank 3: `majkel_near_1` (100 episodes)
- `bono` rank 4: `majkel_near_1` (100 episodes)
- `LiamK` rank 7: `alakazam_no_enriching_boss_fez_no_shaymin_dunsparce4-3_candy4_hammer4` (100 episodes)
- `Rmy` rank 9: `majkel_near_1` (100 episodes)
- `THIRD PTCG Club` rank 15: `majkel_near_3` (100 episodes)
- `matsurih` rank 16: `majkel_exact` (100 episodes)
- `ei ei ei yikuso` rank 17: `majkel_near_1` (100 episodes)

## 3. Used games and decisions

162 normal complete Majkel games, 11438 decisions, and 89729 legal candidates.

## 4. Excluded data and reasons

- `replay_missing_observation_action`: 1956
- `duplicate_episode`: 43
- `abnormal_end`: 2

## 5. Action alignment

Used `observation[t] -> action[t+1]`: legal rate across all stored step actions 54.14%, versus 41.80% at the same step. For non-empty recorded actions, next-step legality is 100.00%; remaining mismatches are inactive-seat empty actions paired with mandatory selections.

## 6. Leakage prevention

Acting observations only; opponent private hand IDs ignored; visualize/full decks limited to audit metadata; outcome/future events limited to labels and weights; episode-level splits.

## 7. Teacher weights

Rank/outcome base x data quality x decision importance x post-action quality x repeated tactical-state agreement, clipped to [0.05, 2.0]. Losses remain with lower weight.

## 8. Model

LightGBM LambdaRank candidate scorer (76 scalar features) plus a separate LightGBM value head. A 48/24-unit MLP legal-softmax baseline was also trained. The winning tree model is distilled exactly to pure-Python JSON.

## 9. Offline evaluation

Exact Top 1 72.40%, semantic Top 1 74.17%, Top 3 90.59%, Top 5 96.14%, MRR 0.833, weighted log loss 1.094, ECE 0.222.

## 10. Action-type results

- `ability`: n=111, Top 1 92.62%, Top 3 98.90%
- `attack`: n=139, Top 1 88.26%, Top 3 98.45%
- `boss`: n=22, Top 1 24.26%, Top 3 64.11%
- `card`: n=889, Top 1 71.10%, Top 3 90.26%
- `end`: n=52, Top 1 48.44%, Top 3 69.47%
- `energy`: n=119, Top 1 44.71%, Top 3 69.14%
- `evolve`: n=227, Top 1 85.60%, Top 3 99.55%
- `hammer`: n=27, Top 1 57.26%, Top 3 96.26%
- `no`: n=17, Top 1 0.00%, Top 3 100.00%
- `none`: n=56, Top 1 100.00%, Top 3 100.00%
- `number`: n=6, Top 1 100.00%, Top 3 100.00%
- `retreat`: n=15, Top 1 7.74%, Top 3 30.32%
- `xerosic`: n=35, Top 1 48.57%, Top 3 62.78%
- `yes`: n=150, Top 1 100.00%, Top 3 100.00%

## 11. Ablations

- `unweighted`: Top 1 71.25%, Top 3 89.22%
- `wins_only`: Top 1 68.77%, Top 3 88.78%
- `exclude_unique_legal`: Top 1 70.51%, Top 3 89.00%
- `no_post_action_quality`: Top 1 71.12%, Top 3 88.80%
- `no_agreement`: Top 1 70.80%, Top 3 89.47%
- `rank_weight_on_off`: not_identifiable (all trainable episodes are rank-1)
- `majkel_vs_multiple_top`: not_available (other bundles lack observations/actions/legal candidates)
- `deck_type_input_on_off`: not_identifiable (all trainable episodes use one exact deck)
- `value_head_on_off`: reported_separately (see report)
- `safety_rules_on_off`: deferred_to_golden_and_battle_evaluation (see report)
- `fallback_on_off`: deferred_to_hybrid_evaluation (see report)

## 12. Battle evaluation

Native shuffle seeds cannot be fixed; seats were alternated and Wilson intervals are reported. No Rating improvement is claimed.

- `alakazam741_v9_top8_core`: 40 games, win rate 15.00% (95% CI 7.06%-29.07%), crashes 0, illegal selections 0
- `alakazam741_v11_board_depth`: 40 games, win rate 32.50% (95% CI 20.08%-47.98%), crashes 0, illegal selections 0
- `alakazam741_v12_top_sync_full`: 40 games, win rate 50.00% (95% CI 35.20%-64.80%), crashes 0, illegal selections 0

### Opponent pool

- `Alakazam generic`: 30 games, win rate 80.00% (95% CI 62.69%-90.50%), crashes 0, illegal selections 0
- `Crustle`: 30 games, win rate 90.00% (95% CI 74.38%-96.54%), crashes 0, illegal selections 0
- `Grimmsnarl`: 30 games, win rate 83.33% (95% CI 66.44%-92.66%), crashes 0, illegal selections 0
- `Mega Kangaskhan`: 30 games, win rate 96.67% (95% CI 83.33%-99.41%), crashes 0, illegal selections 0
- `Mega Starmie`: 30 games, win rate 53.33% (95% CI 36.14%-69.77%), crashes 0, illegal selections 0
- `Team Rocket Spidops`: 30 games, win rate 6.67% (95% CI 1.85%-21.32%), crashes 0, illegal selections 0

Threshold ablation against v12: 0.58, 0.65, and 0.75 each scored 50.00% over 40 games; pure fallback at 1.10 scored 52.50%. All confidence intervals overlap. Threshold 0.65 was retained as the lowest conservative setting without a measured safety regression.

## 13. v9, v11, and top reconstruction comparison

At threshold 0.65 the hybrid scored 15.00% against v9, 32.50% against v11, and 50.00% against the exact-deck v12 fallback over 40 games per pair. The v9/v11 agents use different decks, so these are operational comparisons rather than controlled policy-only ablations. Gate 3 remains partial.

## 14. Submission compatibility

Runtime uses Python standard library only. LightGBM, NumPy, pandas, PyTorch, and scikit-learn are not runtime dependencies. The local repository lacks redistributable official `cg/`; the payload must be combined with competition-provided official `cg/` on Kaggle.

## 15. Tests

54 ML/golden checks passed, including 42 isolated v12 safety states, data leakage checks, exact tree distillation, missing model, NaN, and timeout fallback. Local battles recorded zero illegal actions and crashes. The pre-existing repository suite still has one unrelated v10 failure: `test_battle_cage_does_not_lose_current_ko`; it also fails when run without `ml_alakazam` tests.

## 16. Created files

All new implementation and artifacts are under `ml_alakazam/`; existing agents were not modified.

- `ml_alakazam/README.md`
- `ml_alakazam/agents/alakazam_ml_v1/README.md`
- `ml_alakazam/agents/alakazam_ml_v1/common_runtime.py`
- `ml_alakazam/agents/alakazam_ml_v1/deck.csv`
- `ml_alakazam/agents/alakazam_ml_v1/fallback_v12.py`
- `ml_alakazam/agents/alakazam_ml_v1/main.py`
- `ml_alakazam/agents/alakazam_ml_v1/metadata.json`
- `ml_alakazam/agents/alakazam_ml_v1/ml_features.py`
- `ml_alakazam/agents/alakazam_ml_v1/ml_runtime.py`
- `ml_alakazam/agents/alakazam_ml_v1/policy_base.py`
- `ml_alakazam/agents/alakazam_ml_v1/ranker_model.json`
- `ml_alakazam/configs/default.json`
- `ml_alakazam/configs/training.json`
- `ml_alakazam/data_processed/dataset_stats.json`
- `ml_alakazam/data_processed/decision_dataset.parquet`
- `ml_alakazam/data_processed/deck_clusters.csv`
- `ml_alakazam/data_processed/episode_manifest.csv`
- `ml_alakazam/data_processed/expert_weights.parquet`
- `ml_alakazam/data_processed/legal_candidate_dataset.parquet`
- `ml_alakazam/data_processed/manifest_stats.json`
- `ml_alakazam/models/neural_ranker.json`
- `ml_alakazam/models/ranker_model.json`
- `ml_alakazam/models/ranker_model.txt`
- `ml_alakazam/models/value_model.json`
- `ml_alakazam/models/value_model.txt`
- `ml_alakazam/reports/ablation_results.json`
- `ml_alakazam/reports/action_type_metrics.json`
- `ml_alakazam/reports/alignment_report.json`
- `ml_alakazam/reports/battle_evaluation.json`
- `ml_alakazam/reports/battle_smoke.json`
- `ml_alakazam/reports/dataset_audit.md`
- `ml_alakazam/reports/feature_spec.md`
- `ml_alakazam/reports/final_report.md`
- `ml_alakazam/reports/lightgbm_decision_results.csv`
- `ml_alakazam/reports/offline_evaluation.json`
- `ml_alakazam/reports/offline_evaluation.md`
- `ml_alakazam/reports/reproducibility.json`
- `ml_alakazam/reports/test_predictions.csv`
- `ml_alakazam/reports/threshold_ablation.json`
- `ml_alakazam/src/__init__.py`
- `ml_alakazam/src/build_dataset.py`
- `ml_alakazam/src/build_manifest.py`
- `ml_alakazam/src/common.py`
- `ml_alakazam/src/distill_model.py`
- `ml_alakazam/src/evaluate_battle.py`
- `ml_alakazam/src/evaluate_offline.py`
- `ml_alakazam/src/evaluate_thresholds.py`
- `ml_alakazam/src/export_submission.py`
- `ml_alakazam/src/feature_engineering.py`
- `ml_alakazam/src/parse_replays.py`
- `ml_alakazam/src/replay_io.py`
- `ml_alakazam/src/reporting.py`
- `ml_alakazam/src/run_pipeline.py`
- `ml_alakazam/src/train_policy.py`
- `ml_alakazam/src/train_ranker.py`
- `ml_alakazam/tests/golden_states_suite.py`
- `ml_alakazam/tests/test_data_pipeline.py`
- `ml_alakazam/tests/test_golden_subprocess.py`
- `ml_alakazam/tests/test_model_runtime.py`

## 17. Submission ZIP

- `ml_alakazam/artifacts/alakazam_ml_v1_payload.zip`
- `ml_alakazam/artifacts/ml_alakazam_complete.zip`
- `ml_alakazam/artifacts/export_manifest.json`
- `ml_alakazam/artifacts/SHA256SUMS.txt`

## 18. Remaining concerns

Only one teacher submission has full observations/actions/legal candidates; calibration is weak; rare Boss/energy/retreat decisions are below deployment quality; exact seeded pairing and official engine submission validation are unavailable locally; Gate 3 is not conclusively passed.

## 19. Next improvements

Acquire full replays for the other seven Alakazam teams, retrain team/submission/deck holdouts, calibrate on a separate split, add official-engine seeded evaluation if exposed, and target Spidops before any reinforcement learning or deck change.
