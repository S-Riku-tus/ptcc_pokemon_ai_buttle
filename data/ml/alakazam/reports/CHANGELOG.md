# Change log

- Fixed plural `replays/episode_*.json` discovery while retaining singular `replay/`.
- Added conservative target-seat inference and exclusion reasons.
- Added deck hashes, Majkel substitution distance, named deck differences and clusters.
- Replaced log-only action alignment with exact same-seat replay action-index alignment.
- Expanded leakage-safe features from 143 to 225 and moderated teacher weights.
- Added team, submission and deck holdouts, calibration, ablation and corpus comparison.
- Added safe hybrid thresholds and forced fallback for weak focus actions.
- Added replay-policy smoke tests and an integration-only exported runtime.
