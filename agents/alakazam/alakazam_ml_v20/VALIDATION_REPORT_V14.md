# VALIDATION REPORT V14

## Completed

- 60-card deck and one ACE SPEC retained from v13
- Python compilation: pass
- Runtime/model import: pass
- Self-contained regression suite: 53 passed
- Saved v13 ladder audit: 52 games parsed

New Golden states cover:

- protected Powerful Hand beats END but remains below productive actions;
- Psychic-to-Active attack fuel beats Rich-cycle attachment;
- Boss escapes Mist lock for a low-value unprotected Bench KO;
- all Hammers are reserved after Mist is seen and another copy is likely;
- attached Mist immediately releases the reservation and is hammered.
- Rich draw-stop backup ETA is defined and no longer falls back at runtime.

## Local engine smoke comparison

Balanced-seat runs in the repository's local official-compatible engine:

| Opponent | Games | v14 wins | Win rate |
|---|---:|---:|---:|
| v13 | 160 | 81 | 50.625% |
| v11 | 160 | 70 | 43.750% |

Across all 320 games: zero crashes, zero illegal selections, and zero policy
fallbacks. The v11 result agrees with the user's view that v11 remains a strong
reference; v14 is not presented as proven stronger before ladder evaluation.
The native engine is not fully seed-reproducible, so these runs are primarily a
runtime-safety smoke test rather than a promotion claim.

## Not yet measured

- Kaggle validation episode;
- v14 ladder rating.

The changes directly close observed v13 decisions, but no unsubmitted candidate
can honestly guarantee a rating improvement before engine and ladder testing.
