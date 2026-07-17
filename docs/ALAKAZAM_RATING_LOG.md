# Alakazam Rating Log

Last updated: 2026-07-17 JST

## Final Ratings

| Agent | Final rating | Notes |
| --- | ---: | --- |
| `alakazam741_v1` | 845.7 | Strong baseline. |
| `alakazam741_v2` | 855.4 | Highest final rating among the recorded versions. |
| `alakazam741_v3` | 850.3 | Close to v2, broadly same strength band. |
| `alakazam741_v4` | 724.8 | Large regression in real ladder. |
| `alakazam741_v5` | 763.5 | Partial recovery from v4, but still clearly below v1-v3. |
| `alakazam741_v6` | 700.2 | Regression from v5 and below v4 in the final ladder result. |
| `alakazam741_v8` | 675.7 | Further regression versus v6 and the lowest recorded ladder result so far. |

## ML Final Ratings

| Agent | Final rating | Notes |
| --- | ---: | --- |
| `alakazam_ml_v1` | N/A | Not submitted / no final ladder rating recorded. |
| `alakazam_ml_v2_expanded` | 691.7 | ML-series ladder result. Keep separate from the deterministic `alakazam741_v*` rating line. |

## Interpretation

v1-v3 are in the same practical strength band, while v4, v5, v6, and v8 remain clearly below that tier.
Compared with v2, v4 is lower by 130.6 rating points, v5 is lower by 91.9, v6 is lower by 155.2, and v8 is lower by 179.7.
Compared with v3, v4 is lower by 125.5 rating points, v5 is lower by 86.8, v6 is lower by 150.1, and v8 is lower by 174.6.

This changes the v5 direction:

- Do not use v4 as the base unless there is a very specific reason.
- Prefer `alakazam741_v2` or `alakazam741_v3` as the v5 base.
- Treat v4 primarily as a failure-analysis source.
- Port only the confirmed P0 lessons from v4 logs, especially:
  - prevent solo Dudunsparce / Run Away Draw self-loss,
  - detect effect-prevention states such as Mist Energy, Rock Fighting Energy, and Team Rocket's Articuno,
  - avoid Powerful Hand into effect-prevented targets,
  - tighten optional draw and search when the deck is low.

## Working Conclusion For v5

The safest v5 direction is:

`v5 = v2/v3 strong skeleton + v4 log-confirmed P0 bug fixes + small tech-card A/B tests`

The v4 local self-play results should be treated as overfit to older Alakazam variants, not as reliable evidence of real ladder strength.

## Takeaway After v5/v6 Results

v5 recovered some rating versus v4, but not enough to rejoin the v1-v3 band.
v6 then regressed to 700.2, which is 63.3 below v5 and 24.6 below v4.
That supports treating the v4/v5-derived fixes as selective lessons rather than as a broad new base direction.

## Takeaway After v8 Result

v8 finished at 675.7, which is 24.5 below v6, 48.1 below v4, and 179.7 below v2.
That keeps the current conclusion unchanged: use v2 or v3 as the base, and treat later low-rated variants mainly as bug-fix and failure-analysis inputs rather than new foundations.

## Takeaway After ML v2 Result

`alakazam_ml_v2_expanded` finished at 691.7. Since `alakazam_ml_v1` was not submitted, there is no v1-to-v2 ladder comparison inside the ML series.
The result should be interpreted separately from the deterministic `alakazam741_v*` line: the current ML runtime needs failure analysis and guarded/retrained follow-up before it can replace the stronger deterministic baselines.
