# Grimmsnarl ML v3

`grimmsnarl_ml_v3` keeps the deployed v2.1 candidate ranker and adds a small
next-action prior distilled from the five strongest same-list Grimmsnarl
teachers in the corpus (leaderboard ranks 4, 5, 9, 11 and 13).

## Policy

1. The 919-tree v2.1 ranker scores each legal semantic candidate. It remains
   pinned to teacher 16494330 and retains the v2.1 wall, effective-damage,
   Boss, damage-counter and turn-history fixes.
2. A decision-level multiclass model reads public state, the complete legal
   action-family menu and summaries of the v2 scores.
3. Candidate scores are combined as `zscore(v2) + 0.10 * action_logit`.
   Therefore the prior can adjust whether to evolve, attach, use an ability,
   attack, and so on, while v2 still chooses the concrete card or target.
4. Multi-pick and unmeasured choices retain the v7 rule fallback.

The coefficient is intentionally conservative. Validation selected it by
maximising agreement with the five elite teachers while allowing at most a
0.60-point loss against the existing pinned teacher.

## Main validation

- Elite chronological test: 75.84% to 76.37% (`+0.53pt`, n=9,792).
- Pinned-teacher chronological test: 91.96% to 91.82% (`-0.14pt`, n=1,480).
- Real runtime on rank-4 replays: all-context 67.11% to 67.72%; MAIN 60.01%
  to 61.12%.
- Real runtime on the pinned teacher: all-context 91.34% to 91.20%.
- Paired self-play versus v2.1: 35-25 over 60 games, with zero errors,
  illegal selections or fallbacks.
- Replay runtime: about 34 ms mean and 66 ms p95.

See `VALIDATION_REPORT_V3.md` and
`experiments/grimmsnarl_ml_v3/RESULTS.md` for the selection and rejected
ablations. Ladder performance is not claimed until this directory is run as
a separate Kaggle submission.

## Files

- `ranker_model.json`: unchanged v2.1 candidate ranker.
- `action_model.json`: 15-class elite action prior, 143 trees per class.
- `ml_runtime.py`: standard-library two-stage inference and safe v2 fallback.
- `ml_features.py`: unchanged v2.1 public-state and candidate features.
- `fallback_policy.py`: inherited v7 policy for choices outside ML scope.

