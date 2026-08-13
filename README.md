# PTCG AI Battle Agent

Repository for Pokemon TCG AI Battle Challenge agents.

## Current Agents

- `agents/alakazam_ml_v3`: **current Champion**. Guarded ML hybrid over the v12 fallback (adds safety gates to the ML runtime).
- `agents/alakazam741_v12_top_sync_full`: deterministic Alakazam fallback policy.
- `agents/alakazam_ml_v2_expanded`: earlier ML hybrid agent. Keeps the v12 fallback and uses a distilled LightGBM candidate ranker only for high-confidence ACTIVE MAIN decisions.

Older Alakazam versions remain under `agents/` as history and regression references. Non-Alakazam reconstruction work is kept in its own agent directory or under local archives.

## Layout

```text
agents/
  alakazam741_v12_top_sync_full/
  alakazam_ml_v2_expanded/
ml/
  core/                    # replay, dataset, feature, split, train, evaluation helpers
  archetypes/              # deck-specific plugins such as Alakazam
  pipelines/               # one-command training entry points
  configs/
data/ml/alakazam/
  processed/
  models/
  reports/
kaggle/
scripts/
tests/
```

`agents/` contains Kaggle runtime files only. Training datasets, reports, joblib models, and notebooks do not belong inside agent directories.

## Setup

Runtime submissions use only the Python standard library plus the official competition `cg/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Use `requirements-ml.txt` only for retraining/evaluation. It is not needed for Kaggle submission runtime.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent alakazam_ml_v2_expanded
```

## Retrain Or Re-Export

Full pipeline from replay ZIPs, when available:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam
```

Reuse existing processed data:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed
```

Raw replay ZIPs are expected under `data/runs/kaggle_top50/` by default. Paths are configurable in `ml/configs/alakazam.json`.

## Build Kaggle Submission

Official submission archives are tar.gz files built only by `scripts/build_submission.py`.

```powershell
.\.venv\Scripts\python.exe .\scripts\build_submission.py `
  --agent alakazam_ml_v2_expanded `
  --cg-source <official cg path> `
  --output submission_alakazam_ml_v2_expanded.tar.gz
```

The builder validates `main.py`, `deck.csv`, `cg/api.py`, and rejects training artifacts such as datasets, reports, and joblib files.

On Kaggle Notebook:

```powershell
python kaggle/create_submission_from_git.py
```

This builds both `alakazam741_v12_top_sync_full` and `alakazam_ml_v2_expanded`.

## More Docs

- `docs/ML_ARCHITECTURE.md`
- `docs/ML_REPRODUCTION.md`
- `docs/ML_MULTI_ARCHETYPE_ROADMAP.md`
- `docs/ML_MIGRATION_REPORT.md`
