# ML Alakazam Pipeline

This directory contains the training and evaluation code for the
`alakazam_ml_v2_expanded` hybrid agent.

The official Kaggle runtime lives outside this package:

```text
agents/alakazam_ml_v2_expanded/
```

Training artifacts live under:

```text
data/ml/alakazam/
  processed/
  models/
  reports/
```

## Reproduce

Install the ML dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Run the full pipeline from replay ZIPs when they are present:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam
```

Reuse the existing processed dataset:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed
```

The pipeline updates `agents/alakazam_ml_v2_expanded/ranker_model.json` only
after writing the model under `data/ml/alakazam/models/`. Existing runtime models
are archived under `data/ml/alakazam/models/archive/` before being replaced.

## Submission

The ML pipeline does not create Kaggle archives. Use:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_submission.py `
  --agent alakazam_ml_v2_expanded `
  --cg-source <official cg path> `
  --output submission_alakazam_ml_v2_expanded.tar.gz
```

The legacy development ZIP exporter remains under `ml/legacy/src/export_submission.py`
only for historical inspection. Its output is not Kaggle-submittable.

