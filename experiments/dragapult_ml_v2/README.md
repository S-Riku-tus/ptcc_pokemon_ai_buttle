# Dragapult ML v2 reproduction

Target list `202ee2cec6cbe8b4`. Teacher cohort: the nine exact-list submissions
from the 2026-08-14 top-40 snapshot plus the eight found in the 2026-08-16
top-50 probe (`teachers_20260816.csv`, 17 submissions / 15 distinct teams).

## 0. Which decks are worth imitating right now

```powershell
.\.venv\Scripts\python.exe scripts\fetch_kaggle_top100_snapshot.py `
  --top-n 100 --output-root .tmp\lb_20260816

.\.venv\Scripts\python.exe experiments\dragapult_ml_v1\probe_field.py `
  --submissions .tmp\lb_20260816\latest\public_submissions_top100.csv `
  --scratch .tmp\deck_probe_20260816 --top 60 `
  --out experiments\dragapult_ml_v1\field_20260816.json
```

One replay per representative submission; nothing is written into the corpus.
Kaggle answers HTTP 429 after roughly 60 listings, so probe in blocks.

## 1. Collect and verify

```powershell
.\.venv\Scripts\python.exe scripts\collect_exact_deck_teachers.py `
  --teachers experiments\dragapult_ml_v2\teachers_20260816.csv `
  --deck-hash 202ee2cec6cbe8b4 `
  --output-root data\kaggle_dragapult_exact `
  --refresh-lists --workers 4 --list-retries 6 --retry-delay 8
```

`--refresh-lists` matters: the EpisodeService keys episodes by submission id,
and 6 of the 8 current exact-list teams were new teams under new submission
ids. Result: 1,392 verified trajectories, 15 teachers, 0 deck mismatches,
0 seat errors.

## 2. Card tables, corpus, model

```powershell
.\.venv\Scripts\python.exe scripts\build_dragapult_card_tables.py `
  --targets agents\dragapult\dragapult_ml_v2\ml_features.py

.\.venv\Scripts\python.exe scripts\build_grimmsnarl_v2_corpus.py `
  --data-root data\kaggle_dragapult_exact `
  --agent-dir agents\dragapult\dragapult_ml_v2 `
  --deck-hash 202ee2cec6cbe8b4 `
  --selection-manifest-out experiments\dragapult_ml_v2\selected_episodes_20260816.csv `
  --output data\ml\dragapult_v2\corpus_full.npz `
  --report experiments\dragapult_ml_v2\corpus_full.json --workers 10

.\.venv\Scripts\python.exe scripts\train_grimmsnarl_v2_teacher.py `
  --corpus data\ml\dragapult_v2\corpus_full.npz `
  --team-feature --split-mode per-team `
  --episode-equal-weight --teacher-equal-weight `
  --num-boost-round 5000 --early-stopping 800 --threads 18 `
  --output-model data\ml\dragapult_v2\ranker_full.txt `
  --report experiments\dragapult_ml_v2\train_full.json

.\.venv\Scripts\python.exe scripts\build_dragapult_discard_table.py `
  --data-root data\kaggle_dragapult_exact `
  --split-report experiments\dragapult_ml_v2\train_full.json `
  --target agents\dragapult\dragapult_ml_v2\fallback_policy.py `
  --report experiments\dragapult_ml_v2\discard_table.json

.\.venv\Scripts\python.exe scripts\export_grimmsnarl_v1_model.py `
  --model data\ml\dragapult_v2\ranker_full.txt `
  --corpus data\ml\dragapult_v2\corpus_full.npz `
  --teacher-team 16380946 `
  --report experiments\dragapult_ml_v2\train_full.json `
  --output agents\dragapult\dragapult_ml_v2\ranker_model.json
```

The discard table must be generated *after* training, because it reads the
per-team split boundaries and may only use training episodes.

## 3. Evaluate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_imitation_runtime.py `
  --agent-dir agents\dragapult\dragapult_ml_v2 `
  --data-root data\kaggle_dragapult_exact `
  --split-report experiments\dragapult_ml_v2\train_full.json --split test `
  --report experiments\dragapult_ml_v2\runtime_eval_v2full_newtest.json

.\.venv\Scripts\python.exe scripts\arena_dragapult.py `
  --a agents\dragapult\dragapult_ml_v2 `
  --b data\submissions\submission_55545828_dragapult_v1\submitted_v1_0 `
  --games 60 `
  --report experiments\dragapult_ml_v2\arena_v2_vs_submitted_v1_0.json

.\.venv\Scripts\python.exe scripts\counterfactual_dragapult_v2.py `
  data\submissions\submission_55545828_dragapult_v1 `
  --baseline data\submissions\submission_55545828_dragapult_v1\submitted_v1_0 `
  --candidate agents\dragapult\dragapult_ml_v2 `
  --report experiments\dragapult_ml_v2\counterfactual_live_v2.json
```

### Which bundle is the baseline

`fetch_submission_logs.py --deck-dir` snapshots the *local working tree* at
fetch time into `deck_snapshot/`, which is not the bundle Kaggle ran and drifts
every time the local agent is edited. The reproducible baseline is
`submitted_v1_0/`, extracted from `artifacts/dragapult_ml_v1_preliminary.tar.gz`
and verified against the run itself:

```powershell
tar xzf artifacts\dragapult_ml_v1_preliminary.tar.gz `
  -C data\submissions\submission_55545828_dragapult_v1\submitted_v1_0

.\.venv\Scripts\python.exe scripts\probe_dragapult_live_decisions.py `
  data\submissions\submission_55545828_dragapult_v1 `
  --agent-dir data\submissions\submission_55545828_dragapult_v1\submitted_v1_0 `
  --report experiments\dragapult_ml_v2\live_probe_v1_0_22games.json
```

It reproduces 2,031 of the run's 2,053 decisions (0.9893) and contains no
guard at all, which is how the run's 23 duplicate-colour attachments are
attributed: all 23 came from the ranker, 18 of them with a better route
attachment on offer in the same decision.

## Controlled ablation

`corpus_same_episodes.npz` rebuilds the *identical* 854 episodes of v1 with the
v2 feature module (same selection manifest sha256, same split boundaries, same
75,466 decisions and 462,544 candidate rows), so `train_newfeat.json` versus
`train_v1feat_p800.json` isolates the feature change from the data change and
from the early-stopping change.

## Frozen results

| measurement | v1 | v2 |
|---|---:|---:|
| verified trajectories / teachers | 854 / 9 | 1392 / 15 |
| ranker test Top-1 (own split) | 0.7557 | 0.7462 |
| ranker test Top-3 (own split) | 0.9634 | 0.9634 |
| feature ablation Top-1 (same 854 eps, patience 800) | 0.7568 | 0.7606 |
| shell agreement, 167-episode test split | 0.6862 | **0.7295** |
| MAIN | 0.5657 | 0.6391 |
| Ultra Ball discard | 0.2100 | 0.4484 |
| legal rate / exceptions | 1.0 / 0 | 1.0 / 0 |
| mean / p95 ms | 10.6 / 30.2 | 14.9 / 43.1 |

The two ranker Top-1 numbers are on *different* test splits: v2's split holds
167 episodes from 15 pilots against v1's 102 from 9, and more pilots means more
policy disagreement to predict. The like-for-like number is the ablation row.

60 local games against `submitted_v1_0`, the bundle Kaggle actually ran:

| | v2 | submitted v1.0 | v1.0 on the ladder | teachers |
|---|---:|---:|---:|---:|
| reaches Phantom Dive colours | 0.900 | 0.617 | 0.682 | 0.943 |
| first ready, own-turn mean | 3.72 | 4.24 | 6.40 | 3.69 |
| uses Phantom Dive | 0.883 | 0.617 | 0.636 | 0.940 |
| head-to-head record | 44-16 | | | |

The head-to-head is a mirror on an unseedable shuffle and is not a rating
estimate; the development columns are, because they are one number per game
rather than one bit, and they now sit on the teachers' curve.

An earlier run of this arena reported 51-9 against `deck_snapshot`. That
directory had by then been overwritten with the local v1.1 build, so 51-9 is
v2 against **v1.1**, not against what was submitted. v1.1 also develops worse
than the guardless v1.0 it was meant to improve (0.550 ready at own-turn 5.03
against 0.617 at 4.24) — one more measurement of the same "guards lose" result.

Live counterfactual over the 1,955 single-pick decisions of the real run, with
every agent forced onto the trajectory that was actually played:

| | submitted v1.0 | v2 |
|---|---:|---:|
| duplicate-colour route attachments | 21 | 0 |
| pair-completing route attachments | 24 | 109 |
| agreement with the live action | 0.9898 | 0.8271 |
| duplicates with a completing option on offer | 12 | 0 of those 12 repeated |
