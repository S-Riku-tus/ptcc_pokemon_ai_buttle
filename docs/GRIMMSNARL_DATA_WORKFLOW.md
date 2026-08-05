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
v5 agent or its byte-identical v4 ranker.

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

## Promotion gate

A retrained ranker is a separate challenger. Keep v5 unchanged until the
candidate has been checked on:

- its immutable chronological validation/test blocks;
- the frozen `corpus_v4.npz` benchmark or equivalent stored-board probes;
- per-team and per-context Top-1, especially MAIN, counter placement, and rare
  contexts;
- paired local play against v5/v4; and
- a ladder run, because imitation accuracy alone is not outcome strength.

Collect continuously, but rebuild in batches (for example after 200-500 new
same-deck games or a material leaderboard shift), not after every episode.
