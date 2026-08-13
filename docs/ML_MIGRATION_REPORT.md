# ML Migration Report

## Before

```text
agents/ml_alakazam/
  agents/alakazam_ml_v1/      # actual Kaggle runtime nested under training project
  configs/
  data_processed/
  models/
  reports/
  src/
  tests/
```

This mixed runtime, training code, processed datasets, models, and reports under `agents/`.

## After

```text
agents/alakazam_ml_v2_expanded/
ml/
  core/
  archetypes/
  pipelines/
  configs/
  legacy/
  tests/
data/ml/alakazam/
  processed/
  models/
  reports/
```

The official agent name is `alakazam_ml_v2_expanded`.

## Issues Found And Fixes

- Nested runtime path: copied the working ML runtime to `agents/alakazam_ml_v2_expanded`.
- Training artifacts under `agents/`: moved processed data, models, reports, source, and tests under `data/ml/alakazam/` and `ml/`.
- Old path assumptions: added `ml/core/paths.py` and config-based path resolution.
- Old ZIP export: marked the legacy ZIP exporter as development-only and made `scripts/build_submission.py` the official tar.gz builder.
- Kaggle Notebook helper lagged active agents: updated `kaggle/create_submission_from_git.py` to build v12 fallback and ML hybrid.
- Dependencies were mixed: added `requirements-runtime.txt`, kept dev tooling in `requirements-dev.txt`, and moved ML stack dependencies to `requirements-ml.txt`.

## Large Data Inventory

Large files kept in place, not deleted:

| path | bytes | policy |
| --- | ---: | --- |
| `data/ml/alakazam/processed/matrix/features.npy` | 1,040,321,828 | LFS/external candidate |
| `data/ml/alakazam/processed/decisions.csv` | 27,398,289 | LFS/external candidate |
| `data/ml/alakazam/processed/dataset_rows.csv.gz` | 26,163,142 | LFS/external candidate |
| `data/ml/alakazam/processed/manifest.csv` | 6,772,432 | LFS candidate |
| `data/ml/alakazam/processed/episode_manifest.csv` | 3,021,891 | LFS candidate |
| `data/ml/alakazam/models/ranker.joblib` | 84,068 | LFS/external candidate if it grows |

`.gitattributes` now routes large processed CSV/CSV.GZ/Parquet, matrix files, joblib files, model text files, run ZIPs, and trajectory JSONL toward Git LFS.

Git history was not rewritten. If old large blobs must be purged from history, run an explicit, reviewed `git lfs migrate` or filter-repo workflow outside this automated migration.

## Commands

Retrain:

```powershell
python -m ml.pipelines.train_archetype --archetype alakazam
```

Reuse processed:

```powershell
python -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed
```

Build submission:

```powershell
python scripts/build_submission.py --agent alakazam_ml_v2_expanded --cg-source <official cg path> --output submission_alakazam_ml_v2_expanded.tar.gz
```

## Model Preservation

The existing distilled `ranker_model.json` was copied into the formal runtime agent. The tree structure was not edited manually. Future pipeline runs archive the previous runtime model before replacing it.

## Verification

Executed during the migration:

| command | result |
| --- | --- |
| `python -m pytest` | `147 passed, 1 failed`; the failure is pre-existing/isolated in `tests/test_alakazam741_v10_route_eta.py::test_battle_cage_does_not_lose_current_ko` and reproduces when run alone |
| `python scripts/validate_agent.py --agent alakazam_ml_v2_expanded` | passed; 60-card deck, no warnings |
| `python scripts/build_submission.py --agent alakazam_ml_v2_expanded --cg-source .\tmp_fake_cg --output .\artifacts\submission_alakazam_ml_v2_expanded_test.tar.gz` | passed; 10 unique archive entries |
| `python -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed --skip-training` | passed; reused processed data and validated runtime |
| `python -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed` | passed; rebuilt matrix, trained LightGBM ranker, distilled JSON model, validated runtime |
| `python -m ml.pipelines.train_archetype --archetype alakazam` | passed; rebuilt processed data from 51 replay ZIPs, trained LightGBM ranker, distilled JSON model, validated runtime |
| `python scripts/analyze_deck_archetypes.py --input data\runs\kaggle_top50 --output data\ml\archetype_analysis` | passed; wrote 171 deck clusters |

Full raw replay retraining changed the runtime model hash to
`2f494b027174388877fce9fb615e7a4cd077ad2279bc5511693a494bca526809`.
Previous models were archived under `data/ml/alakazam/models/archive/`.
