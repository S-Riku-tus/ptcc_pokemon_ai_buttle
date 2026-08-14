# Grimmsnarl ML v25

Date: 2026-08-14

## Verdict

`grimmsnarl_ml_v25` is a ladder challenger built from the measured v22
champion. It replaces the old 2,000-tree ranker with a 228-tree ranker learned
from the frozen 21-pilot corpus plus 120 current games from AlphaTCG, the
1095.3-rated same-deck pilot. It does not inherit the v24 Froslass veto.

The strict chronological AlphaTCG holdout contains 14 games and 901 decisions.
On exactly those decisions:

| Measurement | v22 | v25 | Delta |
|---|---:|---:|---:|
| LightGBM strict Top-1 | 77.14% | 86.35% | +9.21 pp |
| Shipped runtime, all contexts | 75.36% | 84.57% | +9.21 pp |
| Shipped runtime, MAIN | 68.57% | 82.04% | +13.47 pp |

The paired, episode-clustered bootstrap 95% interval for the strict Top-1
delta is **+7.23 to +11.23 percentage points**. The interval excludes zero.

This is evidence that v25 reproduces the current stronger pilot materially
better than v22. It is not yet proof of higher ladder Elo; v25 remains a
challenger until a controlled ladder intervention confirms it.

## Why this design

The existing processed corpus is behavioural-cloning data. For every decision
it records which offered option the teacher selected. It does not record a
counterfactual outcome for the options the teacher did not select. Therefore,
turning final wins into a Q target for every candidate would confound action
quality with the state and the teacher that reached it—the same class of error
that invalidated the v24 Froslass lever.

AlphaTCG's observed actions already contain the pilot's multi-turn judgement.
Conditioned imitation transfers that judgement without inventing labels for
unobserved branches. The new pilot is given 3x fitting weight while all other
pilots remain as regularisation. Early stopping uses AlphaTCG's chronological
validation block; the final 14 games are untouched until evaluation.

Explicit search was investigated but is not shipped in v25. The official
Search API can cross the opponent turn, but the previous v7 search stopped at
the opponent turn and the v12 search used a local arithmetic leaf. Neither
measured the attack-exchange problem identified in the ladder logs. Shipping a
new search override before learning and validating a branch-sensitive H2 value
would combine two changes and repeat the unvalidated-intervention mistake.

## Data and training

- AlphaTCG team: `16381823`
- Submission: `55350342`
- Verified games: 120/120
- Verified deck hash: `9714ab5c3996f6cc`
- Extracted AlphaTCG decisions: 9,164
- Extracted AlphaTCG candidate rows: 43,468
- Merged relations: 4,217
- Merged teams: 22
- Merged decisions: 332,139
- Merged candidate rows: 1,604,649
- Objective: LambdaRank
- AlphaTCG fitting weight: 3.0
- Split: per-team chronological 76/12/12
- Best iteration: 228
- Runtime features: 823
- Runtime model size: 5,240,792 bytes

The manifest adapter re-verifies the deck directly from every replay before a
row can enter the corpus:

`scripts/build_grimmsnarl_v25_peer_manifest.py`

Primary reports:

- `corpus_alphatcg.json`
- `corpus_combined.json`
- `train_alphatcg_focus3.json`
- `v22_on_alphatcg_test.json`
- `paired_v25_vs_v22_alphatcg_test.json`
- `runtime_v22_alphatcg_test.json`
- `runtime_v25_alphatcg_test.json`

## Runtime safety decision

The inherited runtime contains a legacy Froslass class escalation. It was
inert in v22 because both the global pin and the escalation used teacher code
0. Moving the global pin to AlphaTCG code 3 would silently reactivate it.
v25 explicitly sets that escalation mode to `off`; otherwise v24's rejected
lever would have leaked back into the candidate.

Non-model mechanics and the 60-card deck remain those of v22. Contexts 5 and 8
stay with the deterministic fallback because the new validation data did not
support ranker ownership there.

## Validation

- Agent unit tests: 165 passed
- Model load error: none
- Planner load error: none
- Shipped teacher code: 3
- Legacy escalation code at runtime: none
- Submission archive entries: 18
- Extracted archive import smoke: passed
- Archive size: 3,100,649 bytes
- Archive SHA-256:
  `cfbbbd561321794d802ca9d3f4474fe3d3fdae638d9b4f136e4c33c0f7abe8ea`

Submission artifact:

`artifacts/grimmsnarl_ml_v25_submission.tar.gz`

## Next intervention

Run v22 as control and v25 as challenger in the same ladder window. Promotion
must be based on opponent-rating-adjusted Elo and the 950+ mirror cell, with
non-mirror non-inferiority. Overall win rate alone is not a promotion metric.

The next search experiment should remain separate from this candidate:

1. Define horizon by complete turns, not raw select count: current turn,
   opponent reply, then own next attack/end (H2).
2. In visible mirrors only, branch the v22/v25 policy's top 2-3 semantic
   candidates and sample 4-8 hidden states consistent with public cards.
3. Evaluate terminal result first, then prize exchange, surviving attacker,
   counterattack readiness, and a calibrated fixed-phase value model.
4. Override only when the lower-confidence value beats the base action and no
   extra prize is conceded.
5. Require zero illegal actions, strict deadline fallback, logged-state parity,
   and paired self-play improvement before a ladder submission.

That experiment can represent the identified multi-turn exchange problem. It
should be promoted only after it beats this v25 baseline, not merely because it
uses more compute.
