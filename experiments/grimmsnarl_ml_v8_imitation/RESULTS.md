# Grimmsnarl imitation accuracy audit

## Decision

Do not replace the v7 production ranker.  The apparent 85.0% figure is the
mean over 21 different teacher policies, with every held-out decision scored
using that row's own teacher code.  It is a useful representation benchmark,
but it is not the fidelity of the deployed agent, which pins one teacher.

The real v7 runtime, including its fallback, planner, Froslass class teacher,
and Petrel/Stamp class teacher, scores **91.27% exact action-index agreement**
on the pinned teacher's 18 chronological held-out games (1,706 single-pick
decisions).  The requested 90% deployment-fidelity gate is therefore met.
The exact-index metric is stricter than semantic agreement because selecting
an interchangeable duplicate at a different raw index is still counted as a
miss.  The decision-level Wilson 95% interval is 89.83-92.51%; 90% is the
measured point-estimate gate, not a claim that its lower confidence bound is
also above 90%.

## Frozen metrics

| evaluation | teacher/data | decisions | Top-1 |
|---|---|---:|---:|
| old offline pooled benchmark | 21 teachers, each row's code | 38,753 | 85.00% |
| old offline deployed pin | team 16494330 | 1,587 | 92.82% |
| **actual v7 runtime** | team 16494330, chronological holdout | **1,706** | **91.27%** |
| current old model | current team 16452116 holdout | 2,782 | 83.93% |
| current old model | current team 16422241 holdout | 2,719 | 79.37% |

The runtime number is in `runtime_v7_teacher16494330.json`.  Teacher forcing
is correct: the candidate answer is not committed, and every stored action is
committed exactly once before the next decision.

## Data refresh

Four current exact-list submissions were collected into the isolated
`data/kaggle_grimmsnarl_v8` archive.  The frozen selection contains 1,014
submission/seat relations, 992 replay files and four teachers.  The extracted
corpus has 76,947 decisions and 375,798 candidate rows:

- team 16452116: 300 newest games;
- team 16422241: 300 newest games;
- team 16561259: 299 games;
- team 16541765: 115 games.

All selected relations have the exact deck hash `9714ab5c3996f6cc`.  Five
separate observation-log extractions failed, but their replay JSON files were
present and readable; the corpus builder reads the replay directly.

## Methods tested and rejected

1. **Single-current-teacher ranker.**  Training only on team 16452116 reduced
   its chronological test Top-1 from the shared model's 84.17% to 83.11%.
   Shared mechanics are useful; teacher isolation is not the answer.
2. **Decision-level offered-set reranker.**  A second model saw the entire
   offered action set, existing score distribution, and action/card/attack
   classes.  It improved validation by 0.20 points but improved test by zero
   (84.17% to 84.17%).
3. **Fresh top-four shared ranker.**  A 1,380-tree model trained on the new
   current corpus scored 82.22% pooled, 83.50% on team 16452116 and 78.71% on
   team 16422241.  It is not promoted.

The high-rated teachers' residual is dominated by same-turn ordering: for
team 16452116 the current model reaches 93.17% if an action the teacher plays
later in the same turn is accepted, but strict Top-1 is 83.93%.  Extra trees,
teacher isolation, a full-set next-action model, and fresh data did not recover
that ordering gap.  Replacing a verified 91.27% deployed policy with one of
these 79-84% candidates would move away from the stated goal.

## Artifacts

- `corpus_v8_current_top4.json` and `current_top4_selection.csv`: frozen data.
- `old_model_on_current_*.json`: policy-drift checks.
- `team16452116_baseline.json`: dedicated-teacher training.
- `team16452116_next_action.json`: offered-set reranker ablation.
- `train_current_top4.json`: fresh shared-ranker training.
- `runtime_v7_teacher16494330.json`: authoritative deployment-fidelity gate.

No Kaggle submission was made and no production model bytes were changed.
