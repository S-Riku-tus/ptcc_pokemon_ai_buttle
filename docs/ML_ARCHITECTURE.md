# ML Architecture

## Runtime Boundary

The official Kaggle ML runtime is:

```text
agents/alakazam_ml_v2_expanded/
```

It contains only runtime files: `main.py`, `deck.csv`, the v12 fallback, runtime feature code, the distilled `ranker_model.json`, and metadata. It does not contain training CSVs, joblib files, reports, or notebooks.

`main.py` keeps the public `agent()` function as the final callable. The ML ranker only runs after the fallback has produced a legal action. Low-confidence decisions, nested selection, multi-select, hard fallback action types, and fallback-confirmed immediate KO remain fallback-controlled.

## Training Boundary

Training code is under:

```text
ml/core/
ml/archetypes/
ml/pipelines/
ml/configs/
```

Artifacts are under:

```text
data/ml/alakazam/processed/
data/ml/alakazam/models/
data/ml/alakazam/reports/
```

`ml.pipelines.train_archetype` is the official retraining entry point. It updates the runtime model only after writing model artifacts under `data/ml/alakazam/models/`; previous runtime models are archived before replacement.

## Path Resolution

`ml/core/paths.py` provides:

- `find_repo_root()`
- `resolve_config_path()`
- `resolve_data_path()`
- `resolve_agent_path()`
- `locate_card_metadata()`

Relative paths are resolved from the repo root or the config location. Kaggle's official `cg/` is used only by `scripts/build_submission.py`; training metadata lookup is separate and expects `cards.json` and `attacks.json`.

## Archetype Separation

Alakazam-specific card IDs and thresholds live in `ml/archetypes/alakazam.py`. Common helpers in `ml/core/` operate on generic deck/core-card inputs where practical. Future archetypes should add a plugin module rather than editing common training code.

Current Alakazam plugin fields include:

- `archetype_name`
- `core_card_ids`
- `reference_team`
- `reference_deck_selection`
- `fallback_agent_dir`
- `runtime_agent_dir`
- `important_card_ids`
- `hard_fallback_action_types`
- `confidence_thresholds`

## Submission Boundary

The ML pipeline does not build Kaggle archives. Official archives are built with:

```powershell
python scripts/build_submission.py --agent alakazam_ml_v2_expanded --cg-source <official cg path>
```

The builder includes official `cg/` and rejects training artifacts.
