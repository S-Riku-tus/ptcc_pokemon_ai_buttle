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

## Deciding whether a refresh is due

Measure it before deciding, with a throwaway leaderboard snapshot and one replay
per submission - read-only, nothing written into the corpus:

```powershell
.\.venv\Scripts\python.exe .\scripts\fetch_kaggle_top100_snapshot.py `
  --top-n 60 --output-root .\.tmp\lb_snapshot

.\.venv\Scripts\python.exe `
  .\experiments\grimmsnarl_ml_v6\measure_refresh_opportunity.py `
  --submissions .\.tmp\lb_snapshot\latest\public_submissions_top60.csv `
  --scratch .\.tmp\deck_probe --top 40 `
  --out .\experiments\grimmsnarl_ml_v6\refresh_opportunity.json
```

It reports which current submissions still play deck hash `9714ab5c3996f6cc`,
which of those are new to the frozen selection, and how many episodes they have.
It also answers a question no corpus metric does: whether the archetype is still
worth training on at all. On 2026-08-06 only 6 of the top 40 played this list,
against 51% of the top 50 four days earlier.
