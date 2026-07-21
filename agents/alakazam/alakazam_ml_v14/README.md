# alakazam_ml_v14

v14 is a deterministic runtime correction built from the submitted v13 agent
(rating 807.1) and all 52 saved public games for submission `54868871`.
The 60-card v13 list, feature extractor, and shadow-only distilled ranker are
unchanged; the policy now treats completing an available attack as a hard
turn-level objective.

## What changed

- A Psychic attachment that enables an immediate Active Alakazam attack has
  priority over Enriching Energy and every competing attachment.
- Enriching Energy cannot consume the once-per-turn attachment while that
  immediate attack route exists.
- END is ranked below an offered Active Alakazam attack or enabling attachment.
- A zero-effect Powerful Hand into Mist/Rock/global protection is still used as
  the final harmless action, restoring v11 behavior instead of idling.
- Boss's Orders may escape a protected Active into any unprotected same-turn
  Bench KO, even when the target is an otherwise low-value Basic.
- After Mist Energy is publicly seen, all likely remaining Enhanced Hammers are
  reserved for Mist until the public-copy estimate is exhausted.
- An attached effect-prevention Energy anywhere on the opposing board makes
  Enhanced Hammer immediately playable; Mist is always the top target.

## Validation

Run the self-contained tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q agents\alakazam\alakazam_ml_v14
```

The saved-log audit is reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_alakazam_v13_attack_gaps.py `
  data\runs\20260721_194408_alakazam_ml_v13_sub54868871
```

See `ANALYSIS_V14.md`, `CHANGELOG_V14.md`, and `VALIDATION_REPORT_V14.md` for
the evidence and remaining limitations.
