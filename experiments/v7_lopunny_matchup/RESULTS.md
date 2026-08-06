# Grimmsnarl ML v7 — add the only statistically supported decision class

Date: 2026-08-06  
Parent: `grimmsnarl_ml_v6`  
Deck/model/planner: unchanged  
Production change: Petrel search class only

## Decision

`grimmsnarl_ml_v7` must descend from v6. Reopening v4, v4.5 or v5 would mix
already-separated changes (ranker refresh, resource search budgets and the v6
Froslass teacher class) and make the next result uninterpretable. Their ladder
ratings are not a sound selector: the three runs faced opponents averaging
967, 843 and 860, and none sampled the current rank-1/2 Dragapult or the
five-slot `a7ee2991` Lopunny/Froslass cell.

v7 retains v6's Froslass class and enables the class v6 pre-registered:

| class | context / trigger | teacher |
| --- | --- | --- |
| Froslass evolve | MAIN / `evolve_froslass=1` | 16371703, code 0 |
| Petrel → Unfair Stamp | TO_HAND / card 1080 offered | 16561259, code 20 |

One teacher code scores the complete legal set inside each argmax. Scores from
two pilots are never compared directly. Every class now has its own offered,
scored, moved and refused-trigger counters.

## Why the Petrel change is admissible

Taking an Unfair Stamp that cannot be played on the same turn is the only open
behaviour whose pilot rate has a significant rating gradient: Spearman rho
`-0.626`, permutation `p=0.0029`. The deployed pin takes it on 71.3% of its
dead offers; team 16561259 takes it on 33.0%.

The three pre-registered teachers were replayed on the same 39 Petrel choices
after fixing teacher-forcing in the evaluator:

| teacher | own dead-Stamp rate | moved | Stamp refusals | other moves |
| --- | ---: | ---: | ---: | ---: |
| 16561259 | 0.330 | 6 | **5** | 1 |
| 16463316 | 0.420 | 6 | 4 | 2 |
| 16531269 | 0.441 | 4 | 4 | **0** |

16561259 gives the largest deployed reduction, is the lowest-rate teacher and
is still a current top pilot. The extra one unrelated move is counted rather
than hidden.

## Evaluation correction

`analyze_grimmsnarl_v3_behaviour.py` described itself as teacher-forced but did
not set `ranker.teacher_forced`. It therefore committed the candidate action
and then the stored action at every scored decision. Later intra-turn features
were not the replay state. The evaluator now suppresses candidate commits and
advances history exactly once with the stored action. It also has an
`--escalation-only` mode: unrelated choices are not run through 2,000 trees,
but every stored choice still advances history.

This changes the inherited v6 Froslass counters from the old 16 moved / 12
refused to the corrected 17 / 13. v7 produces exactly the same corrected
Froslass result, so the new class does not regress v6's class.

Combined corrected probe over v5's 66 stored games:

| class | offers | moved vs pin | refused trigger |
| --- | ---: | ---: | ---: |
| Froslass evolve | 83 | 17 | 13 |
| Petrel Stamp | 39 | 6 | 5 |
| total | 122 | 23 | 18 |

Feature errors: 0. Score errors: 0. Planner errors: 0.

## First local opponent panel member

The corpus builder can now learn from the opposite seat selected by an exact
deck hash. The first and most important member is `a7ee29914c1dce64` (Mega
Lopunny ex + Mega Froslass ex):

- 106 replay/header-verified games, one exact 60-card signature, zero deck
  mismatches;
- 9,233 decisions and 82,734 candidate rows;
- chronological split of 66 train / 20 validation / 20 test games;
- unseen non-forced semantic agreement 56.65%, MAIN Top-1 48.51%, Top-3
  87.65%, variable-count accuracy 97.58%.

That is enough to generate mixed-archetype boards and detect safety failures,
but not enough to treat its arena win rate as the real matchup. Consequently,
no speculative Lopunny-specific planner rule was added. The original report
correctly says that the whole field is near 42% here; imitation alone cannot
invent the missing strategy, and a 56.6%-fidelity opponent cannot validate one.

## Local safety screens

All 164 inherited/new agent tests pass. Static validation passes. Two 20-game
alternating-seat screens on seed 7001 completed with zero crashes, illegal
selections or draws:

| screen | v7 result | interpretation |
| --- | ---: | --- |
| v7 vs v6 | 9-11 | no large regression; far too small to rank a narrow class |
| v7 vs a7ee panel | 14-6 (v6 control 10-10) | safety only; opponent fidelity fails the outcome gate |

Within the direct v7/v6 run, v7 averaged 73.83 ms/move and v6 71.08 ms/move.
The two arenas ran concurrently, so only the within-run ratio is meaningful.

## What remains

1. Do not submit or promote from these 20-game results. First ingest the v6
   ladder run and verify its pre-registered live Froslass/resource gates.
2. v7 is the correct next Grimmsnarl challenger when a new run is desired. Its
   only new production behaviour is the supported Petrel class.
3. Improve the `a7ee2991` opponent with the current top pilots' larger
   646/217/154-game logs, then calibrate it on held-out real Grimmsnarl games.
   Only after that should Boss-target, support-bench or race planner hypotheses
   be compared locally.
4. Build the remaining exact-hash panel members (`6fa64c0e`, `202ee2ce`,
   `0dede7cb`) with the generalized opponent-seat corpus path. Ogerpon remains
   a priced-in structural loss unless a deck change, not a play preference,
   is tested.

No Kaggle submission or promotion was performed.
