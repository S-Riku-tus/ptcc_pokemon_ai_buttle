# alakazam_ml_v5 - top20-trained shadow build 0.5.0

This build combines the v5 60-card deck with the stable, self-contained v3
decision policy.  The top-20 imitation model is loaded and scored at runtime,
but remains in shadow mode because live overrides did not pass the promotion
gate.

## Runtime policy

- `fallback_v3.py` is the authoritative policy and uses this directory's
  `deck.csv`.
- `ranker_model.json` was retrained from all five full replay bundles under
  `data/runs/20260717_kaggle_top20`.
- ML scores only guarded ACTIVE/MAIN bench, evolution, and attack candidates.
- In the default mode, ML records confidence and disagreement diagnostics but
  returns the fallback action.
- `ALAKAZAM_ML_ENABLE_OVERRIDE=1` enables live override only for controlled
  experiments.  It is intentionally off in production.

## Training corpus

- 5 submissions / 5 teams / 2 deck clusters
- 3,158 full replay files
- 3,161 usable expert trajectories
- 168,239 aligned decisions
- 1,923,948 legal candidate rows
- 5 unresolved decisions; alignment rate 99.997%

The replay loader recovers expert seats from each bundle's `submission.json`
and `episodes.json`, including nested episode layouts.  Duplicate trajectory
IDs are removed before training.

## Deck

- Dunsparce 3 / Dudunsparce 3
- Psychic Energy 3
- Fezandipiti ex 1
- Genesect 1 + Lucky Helmet 1
- Maximum Rod is the only ACE SPEC
- 60 cards total

`fallback_v12.py` is retained as an audited alternative and regression-test
fixture.  Its Fezandipiti KO-clock recursion was removed, but it is not the
active production fallback in this build.
