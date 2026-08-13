# Alakazam Rating Log

Last updated: 2026-08-01 JST

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

## Teacher-Imitation Line (ml_v31 onwards)

Updated 2026-08-01 JST. These three runs share the same 60-card deck and the
same v29 safety shell; only the imitation ranker differs.

| Agent | Submission | Final rating | Public record | Win rate | Opponent mean | Win rate vs 800+ |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `alakazam_ml_v31` | 55076863 | 881.2 | 35-33 | 51.5% | 866.1 | 25/58 (43.1%) |
| `alakazam_ml_v32` | 55094510 | 871.9 | 36-28 | 56.2% | 836.0 | 23/49 (46.9%) |
| `alakazam_ml_v33` | 55129390 | **916.9** | **42-29** | **59.2%** | 860.8 | **29/57 (50.9%)** |

`alakazam_ml_v33` is the current champion and the best result recorded for the
Alakazam line.

### How To Read These

Do not rank versions on the headline rating. The identical `alakazam_ml_v20`
agent scored 842.8 and 804.0 on two runs, so ~40 points at n≈60 is inside the
noise floor, and v33's +35.7 over v31 is inside that band on its own.

Raw win rate has the same defect because matchmaking hands each run a different
pool. Regenerate the opponent-conditioned table instead:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_alakazam_ladder_runs.py `
  --run v31:55076863 --run v32:55094510 --run v33:55129390 `
  --output .\experiments\alakazam_ml_v33\ladder_opponent_buckets.json
```

It re-queries `EpisodeService/ListEpisodes` for each episode's opponent
`initialScore`, which `fetch_submission_logs.py` does not persist.

### The Top-1 Column Above Is The Ranker's, Not The Agent's

Added 2026-08-02. Every Top-1 figure recorded for v31 through v34 scores the
imitation ranker in isolation. The agent does not play the ranker's pick on
16.3% of scoped decisions: the v31 safety shell replaces it with the v29
baseline. Scored on the action actually returned, the v34 holdout reads 77.85%
rather than 83.01%, and all six guards are net negative against the teacher
(`scripts/experiment_alakazam_v35_shell_audit.py`).

That is the most likely reason v34's +5.79 offline gain did not separate from
v33 on the ladder, and v35 is the version that tests it: it narrows the two
worst guards and changes nothing else. From here on, quote the *played*
agreement when comparing versions.

One consequence for the behaviour diagnosis: v34's shell forced attacks so
often that the agent swung 1.6x as much as the teacher on identical boards
(1,481 versus 925 over 9,977 decisions) and played Boss's Orders 2.3x as much.
Powerful Hand count per game is therefore inflated by the guard, and v35 will
read *lower* on it by design. Do not treat that drop as a regression.
