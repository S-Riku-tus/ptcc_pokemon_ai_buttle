# ML Reproduction

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

`requirements-runtime.txt` is intentionally empty except for a note: the Kaggle runtime must not depend on LightGBM, NumPy, or pandas.

## Data Locations

Default raw replay ZIP input:

```text
data/runs/kaggle_top50/*.zip
```

Processed reuse input:

```text
data/ml/alakazam/processed/
```

Model and report outputs:

```text
data/ml/alakazam/models/
data/ml/alakazam/reports/
```

These paths are configurable in `ml/configs/alakazam.json`.

## Commands

Full pipeline from raw replay ZIPs:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam
```

Reuse existing processed data:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed
```

Skip training and only validate/copy the existing distilled model:

```powershell
.\.venv\Scripts\python.exe -m ml.pipelines.train_archetype --archetype alakazam --reuse-processed --skip-training
```

Validate runtime:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_agent.py --agent alakazam_ml_v2_expanded
```

Build Kaggle tar.gz:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_submission.py `
  --agent alakazam_ml_v2_expanded `
  --cg-source <official cg path> `
  --output submission_alakazam_ml_v2_expanded.tar.gz
```

Analyze next deck candidates:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_deck_archetypes.py `
  --input data/runs/kaggle_top50 `
  --output data/ml/archetype_analysis
```

## Safety Notes

- Boss, Retreat, Xerosic, and Hammer remain fallback-controlled.
- Energy is ML-controlled only at the stricter confidence gate.
- Nested/target/multi-select decisions remain fallback-controlled.
- New models are copied to runtime only after training/export and are evaluated from stored or newly collected real ladder episodes.
