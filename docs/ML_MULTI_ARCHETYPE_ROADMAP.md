# ML Multi-Archetype Roadmap

## Goal

The ML stack should support future deck families without putting card IDs, team names, or fallback choices into common code.

Potential future modules:

```text
ml/archetypes/
  alakazam.py
  spidops.py
  lucario.py
  starmie.py
```

## Add A Deck

1. Add `ml/archetypes/<name>.py`.
2. Define core card IDs, reference team/deck selection, important cards, fallback agent, runtime agent, hard fallback action types, and confidence thresholds.
3. Add `ml/configs/<name>.json` with replay input, processed/model/report paths, metadata path, and runtime agent path.
4. Add or copy a runtime agent under `agents/<name>_ml_v1/` with standard-library-only runtime dependencies.
5. Run `scripts/analyze_deck_archetypes.py` on the top replay corpus and verify the deck has enough teams/submissions.
6. Build processed data with `python -m ml.pipelines.train_archetype --archetype <name>`.
7. Validate offline holdouts and action-type safety.
8. Validate behavior on held-out real replay states.
9. Submit explicitly selected candidates and evaluate newly collected ladder episodes.
10. Build Kaggle tar.gz with `scripts/build_submission.py`.

## Current Alakazam Status

Alakazam has:

- 51 replay ZIP bundles in the current local corpus
- 5,122 full replays
- 2,174 target trajectories
- 100,075 aligned decisions
- 1,155,913 legal candidates
- 20 teams
- 21 submissions
- 8 deck clusters

This is enough for the current ML hybrid, but hard fallback and confidence gates remain required.

## Human Review Gates

The deck archetype analyzer emits CSV and Markdown reports. A deck should normally have at least 3 teams before serious ML work; 5 or more teams and enough games is a stronger target. Single-teacher datasets should be treated as imitation of one pilot, not as a robust archetype policy.
