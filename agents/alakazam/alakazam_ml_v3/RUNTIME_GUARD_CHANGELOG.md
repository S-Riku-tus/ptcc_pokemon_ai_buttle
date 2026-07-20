# Runtime guard change log

## Changed

- Moved all abilities back to `fallback_v12`, including Dudunsparce and
  Fezandipiti ex draw abilities.
- Moved END, trainer, energy, disruption, retreat, and unknown actions to
  deterministic fallback.
- Limited ML bench choices to Abra and Dunsparce.
- Forced Fezandipiti ex and Shaymin deployment through their rule-based role
  predicates.
- Preserved a fallback-selected attack against ML development overrides.
- Added candidate-level KO preservation and diagnostic counters.
- Added guarded-runtime unit tests.

## Intentionally unchanged

- `ranker_model.json`
- deck list
- `fallback_v12.py`
- confidence thresholds encoded in the model
- nested selection and legal-option fallback behavior

## Expected effect

The ML adoption rate will fall. This is intentional. The next retrained model
should earn back broader action types only after offline action-type evaluation
and champion–challenger testing show that they are safe.
