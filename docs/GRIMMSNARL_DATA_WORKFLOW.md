# Grimmsnarl Data Workflow

The raw replay collection and a model's training corpus are different assets.
Raw data is append-only; processed corpora and their selection manifests are
immutable, versioned experiment inputs.

## Safety rules

1. Add one relevant submission with `--submission ... --merge-existing`.
2. Never overwrite `corpus_v4.npz`; it is the frozen source of the v4/v5
   ranker and its historical comparison point.
3. Build a new candidate corpus under a new name and write its exact selection
   manifest beside the experiment report.
4. Use only the exact 60-card deck hash `9714ab5c3996f6cc` for this imitation
   line.
5. Cap by team across submission versions. This prevents a prolific pilot or a
   pilot with two collected submissions from dominating the pooled policy.
6. Do not add recency, rating, or outcome weights to a promoted model without
   an A/B run. Those change the learning objective rather than merely fixing
   data validity or balance.

## Append one submission

```powershell
.\.venv\Scripts\python.exe .\scripts\collect_top100_submission_replays.py `
  --input .\data\kaggle_top50_meta\analysis\submissions_grimmsnarl.csv `
  --output-root .\data\kaggle_grimmsnarl_top50 `
  --submission <SUBMISSION_ID> `
  --merge-existing `
  --representative-only `
  --require-deck-card-id 648 `
  --max-episodes-per-submission 300 `
  --newest-episodes-first `
  --workers 1 --sleep 0.1
```

This retains old replay/log files and consolidated index rows. It merges an
episode relation by `(submission_id, episode_id)` and does not run deck-filter
cleanup over untouched data.

## Build a candidate corpus

The recommended current-meta view keeps up to 300 newest relations per team,
across all of that team's submission IDs. `skipped_existing` is included
because the collector validates the replay and both log files before assigning
that status.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_v2_corpus.py `
  --data-root .\data\kaggle_grimmsnarl_top50 `
  --agent-dir .\agents\grimmsnarl\grimmsnarl_ml_v5 `
  --include-skipped-existing `
  --latest-per-team 300 `
  --selection-manifest-out .\experiments\grimmsnarl_ml_v5\data_refresh_selection.csv `
  --output .\data\ml\grimmsnarl\processed\corpus_v5_data_refresh_candidate.npz `
  --report .\experiments\grimmsnarl_ml_v5\corpus_v5_data_refresh_candidate.json `
  --workers 10
```

This command creates a candidate dataset only. It does not modify the deployed
v5 agent or the frozen v4 ranker.

To reproduce the same dataset later even after more logs arrive, build from the
manifest:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_v2_corpus.py `
  --data-root .\data\kaggle_grimmsnarl_top50 `
  --agent-dir .\agents\grimmsnarl\grimmsnarl_ml_v5 `
  --selection-manifest-in .\experiments\grimmsnarl_ml_v5\data_refresh_selection.csv `
  --output .\data\ml\grimmsnarl\processed\corpus_v5_data_refresh_rebuilt.npz `
  --report .\experiments\grimmsnarl_ml_v5\corpus_v5_data_refresh_rebuilt.json `
  --workers 10
```

## Train and compare a challenger

Training is a batch operation, not part of replay collection. On the current
4,097-game corpus the full 4,000-round run took about 43 minutes on this
machine. Keep the deployed model untouched and write a separate challenger:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_grimmsnarl_v2_teacher.py `
  --corpus .\data\ml\grimmsnarl\processed\corpus_v5_data_refresh_candidate.npz `
  --output-model .\data\ml\grimmsnarl\models\ranker_v5_data_refresh_base.txt `
  --report .\experiments\grimmsnarl_ml_v5\train_v5_data_refresh_base.json `
  --team-feature --split-mode per-team --early-stopping 700 --threads 16
```

Compare old and new models on exactly the same test decisions. The optional
iteration cap makes it possible to select an accuracy/runtime Pareto point
without retraining:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_grimmsnarl_ranker.py `
  --corpus .\data\ml\grimmsnarl\processed\corpus_v5_data_refresh_candidate.npz `
  --model .\data\ml\grimmsnarl\models\ranker_v5_data_refresh_base.txt `
  --num-iteration 2000 `
  --baseline-model .\data\ml\grimmsnarl\models\ranker_v4.txt `
  --report .\experiments\grimmsnarl_ml_v5\eval_current_iter2000_vs_v4.json `
  --team-feature --split-mode per-team
```

Export only after the frozen and refreshed comparisons pass:

```powershell
.\.venv\Scripts\python.exe .\scripts\export_grimmsnarl_v1_model.py `
  --model .\data\ml\grimmsnarl\models\ranker_v5_data_refresh_base.txt `
  --num-iteration 2000 `
  --corpus .\data\ml\grimmsnarl\processed\corpus_v5_data_refresh_candidate.npz `
  --teacher-team 16494330 `
  --output .\agents\grimmsnarl\grimmsnarl_ml_v5\ranker_model.json `
  --report .\experiments\grimmsnarl_ml_v5\train_v5_data_refresh_base.json
```

The August 2026 refresh selected 2,000 rather than the 3,515-tree validation
optimum: it kept a clear paired Top-1 gain while reducing the export from 79.2
MB to 45.1 MB. See `experiments/grimmsnarl_ml_v5/DATA_REFRESH_2026-08-06.md`.

## Promotion gate

A retrained ranker is a separate challenger. Keep v5 unchanged until the
candidate has been checked on:

- its immutable chronological validation/test blocks;
- the frozen `corpus_v4.npz` benchmark or equivalent stored-board probes;
- per-team and per-context Top-1, especially MAIN, counter placement, and rare
  contexts;
- paired local play against v5/v4; and
- a ladder run, because imitation accuracy alone is not outcome strength.

Collect continuously, but rebuild in batches, not after every episode. A useful
trigger is at least 200-500 new same-deck games overall, at least 100 new games
for a strategically important top team, a roughly 10% corpus increase, or a
material leaderboard/metagame shift. A trigger starts an experiment; it does
not automatically promote its model.
