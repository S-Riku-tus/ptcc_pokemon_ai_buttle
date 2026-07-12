# Alakazam Rating Log

Last updated: 2026-07-12 JST

## Final Ratings

| Agent | Final rating | Notes |
| --- | ---: | --- |
| `alakazam741_v1` | 845.7 | Strong baseline. |
| `alakazam741_v2` | 855.4 | Highest final rating among v1-v5. |
| `alakazam741_v3` | 850.3 | Close to v2, broadly same strength band. |
| `alakazam741_v4` | 724.8 | Large regression in real ladder. |
| `alakazam741_v5` | 750.5 | Partial recovery from v4, but still clearly below v1-v3. |

## Interpretation

v1-v3 are in the same practical strength band, while v4 and v5 remain clearly below that tier.
Compared with v2, v4 is lower by 130.6 rating points and v5 is lower by 104.9.
Compared with v3, v4 is lower by 125.5 rating points and v5 is lower by 99.8.

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

## Takeaway After v5 Result

v5 recovered some rating versus v4, but not enough to rejoin the v1-v3 band.
That supports treating the v4-derived fixes as selective lessons rather than as a broad new base direction.
