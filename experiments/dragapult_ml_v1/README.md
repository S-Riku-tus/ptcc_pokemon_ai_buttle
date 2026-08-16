# Dragapult ML v1 reproduction

The choice, leakage controls, authority boundary, gates, and next experiment are
recorded in `DESIGN.md`.

Target list: `202ee2cec6cbe8b4`.  The frozen starting cohort is the nine exact
list submissions observed in the 2026-08-14 top-40 snapshot (564 episodes
were advertised at snapshot time).  Every downloaded seat is re-hashed; the
CSV is discovery input, never authority.

```powershell
.\.venv\Scripts\python.exe scripts\collect_exact_deck_teachers.py `
  --teachers experiments\dragapult_ml_v1\teachers_20260814.csv `
  --deck-hash 202ee2cec6cbe8b4 `
  --output-root data\kaggle_dragapult_exact

.\.venv\Scripts\python.exe scripts\build_grimmsnarl_v2_corpus.py `
  --data-root data\kaggle_dragapult_exact `
  --agent-dir agents\dragapult\dragapult_ml_v1 `
  --deck-hash 202ee2cec6cbe8b4 `
  --selection-manifest-out experiments\dragapult_ml_v1\selected_episodes.csv `
  --output data\ml\dragapult_v1\corpus.npz `
  --report experiments\dragapult_ml_v1\corpus_report.json

.\.venv\Scripts\python.exe scripts\train_grimmsnarl_v2_teacher.py `
  --corpus data\ml\dragapult_v1\corpus.npz `
  --team-feature --split-mode per-team `
  --episode-equal-weight --teacher-equal-weight `
  --num-boost-round 4000 --early-stopping 200 --threads 8 `
  --output-model data\ml\dragapult_v1\ranker.txt `
  --report experiments\dragapult_ml_v1\training_report.json

.\.venv\Scripts\python.exe scripts\export_grimmsnarl_v1_model.py `
  --model data\ml\dragapult_v1\ranker.txt `
  --corpus data\ml\dragapult_v1\corpus.npz `
  --teacher-team 16380946 `
  --report experiments\dragapult_ml_v1\training_report.json `
  --output agents\dragapult\dragapult_ml_v1\ranker_model.json

.\.venv\Scripts\python.exe scripts\evaluate_imitation_runtime.py `
  --agent-dir agents\dragapult\dragapult_ml_v1 `
  --data-root data\kaggle_dragapult_exact `
  --split-report experiments\dragapult_ml_v1\training_report.json `
  --split test `
  --report experiments\dragapult_ml_v1\runtime_eval_all_teachers_test.json
```

## Frozen result

- Data integrity: 9 teachers, 854 verified trajectories, 841 unique episodes,
  zero deck mismatches or seat errors.
- Corpus: 75,466 decisions and 462,544 candidate rows.
- Ranker test: top-1 0.7557, top-3 0.9634, order-insensitive top-1 0.8901.
- Submitted-shell chronological test: 102 episodes / 10,612 decisions,
  agreement 0.7327 (mandatory single-pick 0.7573), legal rate 1.0000,
  zero exceptions, mean 10.37 ms and p95 31.359 ms.

The first model remains an offline candidate: 854 is below the 1,000-trajectory
data gate and 0.9634 is below the predeclared 0.9700 top-3 gate.  A later,
untouched batch must also confirm the result.  Search/value learning is
deliberately the next experiment, not mixed into this baseline.
