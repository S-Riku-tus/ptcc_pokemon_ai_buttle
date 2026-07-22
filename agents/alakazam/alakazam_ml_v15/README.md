# alakazam_ml_v15

v15 is a logic/ML successor to v14. The v14 directory is untouched and v15's
60-card deck is byte-for-byte identical to v14, including Enriching Energy.

## What changed

- Ordinary Boss decisions use v11's proven immediate-tempo evaluation instead
  of treating a possible later Active KO as already guaranteed.
- v14's Boss escape from Mist/Rock/global protection remains intact.
- Rich Energy returns to v11's hand-size/Psychic-fuel gate; the hard rule that
  an immediate Active Alakazam attack receives Psychic fuel still remains.
- ML is live only for two reversible, high-holdout-accuracy scopes: choosing
  Abra versus Dunsparce to bench, and choosing the target for the same evolution
  card. Every strategic action class remains deterministic.

## Validation

The finalized-agent rerun went 508-492 against v11 under identical v14 decks,
556-444 against unchanged v14, and 734-66 across four generic opponents. All
2,800 games had zero attackable END selections and zero runtime errors. The
full 58-test suite also passes.

Run the self-contained tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q agents\alakazam\alakazam_ml_v15
```

The saved-log audit is reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_alakazam_v13_attack_gaps.py `
  data\runs\20260721_194408_alakazam_ml_v13_sub54868871
```

See `ANALYSIS_V15.md`, `CHANGELOG_V15.md`, and `VALIDATION_REPORT_V15.md`.
