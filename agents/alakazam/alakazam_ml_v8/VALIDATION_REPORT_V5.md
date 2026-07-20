# Alakazam ML v5 validation report

Date: 2026-07-18

## Result

The completed v5 beats `alakazam741_v3` 567-433 (56.7%) over 1,000 games with
alternating seats and independent per-game seeds.  The Wilson 95% interval is
53.6%-59.7%.  There were no agent errors, illegal actions, policy fallbacks, or
observation fallbacks.

The trained model remains in shadow mode.  Enabling all guarded model
overrides on the same seed schedule produced 109-91 (54.5%), so live override
did not pass the promotion gate.

## Root cause and logic repair

The original v5 result was 46-154 (23.0%).  Disabling ML improved it only to
58-142 (29.0%), and diagnostics exposed 285 internal `RecursionError`
fallbacks.  The cycle was:

`_turns_to_win -> _ko_active_reachable -> _achievable_hand -> _fez_draw_needed -> _turns_to_win`

`fallback_v12.py` now uses a conservative non-recursive prize clock inside
Fezandipiti draw gating.  That removed the recursion and raised the old
fallback result to 67-133 (33.5%).

Deck/policy ablation then showed that the decisive weakness was still the
policy: v3 logic using the unchanged v5 deck reached 124-76 in the diagnostic
harness, while changing only to a top-player deck did not solve the problem.
The active v5 runtime therefore embeds the stable v3 policy in
`fallback_v3.py` and uses the v5 deck.

## Replay ingestion and training

All five `*_full.zip` bundles under `data/runs/20260717_kaggle_top20` were
ingested.  The loader now reads nested `submission.json` and `episodes.json`
metadata to recover the exact expert seat and deduplicates trajectory IDs.

- 5 teams / 5 submissions / 2 deck clusters
- 3,158 full replay files
- 3,161 usable expert trajectories
- 168,239 aligned ACTIVE/MAIN decisions
- 1,923,948 legal candidate rows
- 5 unresolved decisions
- 99.997% alignment rate

Time-holdout imitation metrics were top-1 59.13%, top-3 84.71%, MRR 73.14%,
and high-confidence accepted top-1 76.58%.  Team and deck holdouts were weaker,
which correctly warned that imitation accuracy alone was not sufficient for
live promotion.

## ML battle ablation

Attack-only, bench-only, evolution-only, ability-only, and higher-confidence
override policies were screened.  None produced a stable improvement over the
deterministic policy.  In the final paired comparison, the model changed 136
decisions and reduced wins from 113 to 109.

Default runtime behavior therefore still loads and scores the model, records
confidence and counterfactual disagreements, and returns the deterministic
action.  `ALAKAZAM_ML_ENABLE_OVERRIDE=1` is an experiment-only switch.

## Generalization check

Against five generic non-v3 opponents, 60 independently seeded games each, v5
produced 253 wins and 47 losses (84.3%).  Under the same schedule v3 produced
246 wins and 54 losses (82.0%).  V5 still trailed on Crustle and Kangaskhan but
gained on generic Alakazam, Grimmsnarl, and Mega Starmie, so the logic change is
not a v3-only branch.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\self_play.py alakazam_ml_v5 alakazam741_v3 `
  --games 200 --seed 741 --reseed-each-game --quiet `
  --output-dir data\runs\local_self_play\ml_v5_shadow_vs_v3

$env:ALAKAZAM_ML_ENABLE_OVERRIDE = "1"
.\.venv\Scripts\python.exe scripts\self_play.py alakazam_ml_v5 alakazam741_v3 `
  --games 200 --seed 741 --reseed-each-game --quiet `
  --output-dir data\runs\local_self_play\ml_v5_override_vs_v3
Remove-Item Env:\ALAKAZAM_ML_ENABLE_OVERRIDE
```
