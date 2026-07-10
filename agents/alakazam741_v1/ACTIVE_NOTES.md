# alakazam741_v1 Active Notes

## Role

Strong Alakazam baseline. Keep this agent active as the control deck while improving v2/v3.

This version is useful because it exposes whether a new candidate is truly better, or merely fixes one ladder failure while creating another.

## Known Results

- Ladder submission: `54520948`
- Saved run: `data/runs/20260710_181918_alakazam741_v1_sub54520948`
- Local A/B before v2: v2 beat v1 36-24 over 60 games, but ladder quality looked similar enough to keep v1 as a control.

## Current Use

Use v1 for:

- A/B tests against `alakazam741_v2` and later candidates.
- Regression checks for deck-out prevention, resource timing, and prize-race behavior.
- Comparing ladder failure patterns against v2.

Do not pair this with Garchomp as an active submission line anymore. The current active pair is:

- `alakazam741_v1`
- `alakazam741_v2`

## Next Analysis Questions

- Which v1 losses did v2 actually fix?
- Which v2 losses are new regressions?
- Which failures are common to both and should define v3?
