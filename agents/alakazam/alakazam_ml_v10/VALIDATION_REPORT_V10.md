# v10 validation report

## Ladder failure audit

- Public v9 games: 96 (55 wins / 41 losses).
- Attack opportunity conversion: 100%.
- Deck-out losses: 9; board-out losses: 7.
- Diagnosis: future board/deck runway, not failure to choose an offered attack.

## ML holdouts versus v8 ranker

| Holdout | v8 Top-1 | v10 Top-1 | Delta |
|---|---:|---:|---:|
| time | 60.689% | 60.217% | -0.473pt |
| team | 58.040% | 59.028% | +0.987pt |
| submission | 60.200% | 60.063% | -0.137pt |
| deck | 58.834% | 58.434% | -0.400pt |

Unseen-submission accepted Top-1 improved from 81.142% to 82.589%. Source
balancing improved the time-holdout Top-1 by 0.435pt over the otherwise same
v10 training without source balancing. Mixed transfer results mean shadow-only.

## Decision

Promote the deterministic support-pivot rule and retain the existing deck.
Keep the newly trained ranker in shadow mode. Kaggle rating remains unverified
until a real v10 submission is run.
