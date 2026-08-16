# Dragapult v3 decision record

## Acceptance rule

Dragapult mirror arena results are not used to accept or reject a candidate.
Offline acceptance uses exact-list teacher holdouts and live counterfactuals;
the only local opponent is `agents/grimmsnarl/grimmsnarl_ml_v22`.

## Selected arm

`train_board_focus.json` / `ranker_board_focus.txt`:

- 470 corpus features plus `teacher_team_id`;
- board-width hard-state weight 1.5;
- episode- and teacher-equal weighting;
- teacher pin 16380946;
- test Top-1 0.7455, MAIN 0.6754, order-insensitive 0.8879;
- exported model: 1,088 trees, 24,597,795 bytes.

The packaged runtime result is `runtime_eval_v3_test.json`.  The later guard
survival exception changes the sole guarded teacher decision back to the
teacher's evolution; the focused regression is
`runtime_eval_v3_team16445258_guardcheck.json` (0 guard overrides).

## v22 evidence

`matchup_strategy_grimmsnarl_v22_reference.json` proves that v22's exact deck
hash `9714ab5c3996f6cc` appears in 151 teacher games.  The teachers went 115-36
against that 60-card list.

Selected v3 local files:

- `arena_board_focus_vs_grimmsnarl_v22.json`: 80 games, 24-56;
- `arena_board_focus_vs_grimmsnarl_v22_detailed_40g.json`: 40 additional
  games with censored prizes and Phantom Dive target/counter placement;
- combined: 34-86, mean PD 2.792, P(0/1) 0.200, P(2+) 0.800, P(4+) 0.367,
  max Dragapult 1.642.

The v2 control is `arena_v2_vs_grimmsnarl_v22.json`: 20-60, mean PD 2.450,
P(0/1) 0.263, P(2+) 0.738, P(4+) 0.338, max Dragapult 1.525.  Native shuffle
is unseedable, so the development distributions are stronger evidence than
the records.

## Rejected arms

| arm | teacher result | v22 result | decision |
|---|---|---|---|
| no hard-state focus | test 0.7450, MAIN 0.6740 | not promoted | weaker mechanism |
| board weight 2.0 | test 0.7431, MAIN 0.6734 | 21-59 / 80, P(4+) 0.25 | reject |
| current-pilot weight 1.5 | test 0.7459, MAIN 0.6785, pin 0.7301 | 7-33 / 40, max Pult 1.45 | reject |
| win weight 1.25 | test 0.7470, MAIN 0.6797 | 24-56 / 80, lower PD/P4 than selected | reject |
| pin 16445292 | pooled 0.7220, MAIN 0.6438 | 8-32 / 40 | reject |
| PD4+ episode weight 1.25 | future-conditioned selection | fixed-capacity training did not complete | reject |

## Guard audit

`counterfactual_v3_guard_on_v2.json` replays all 2,450 single-pick live v2
decisions.  The intended bindings are episode 93603391 (Mega Lucario) and
93607548 (Teal Mask Ogerpon).  The first full teacher runtime audit found one
counterexample, episode 93140945: evolving a damaged Active Drakloak makes it
survive a ready opposing Phantom Dive.  The packaged guard now includes only
that measured survival exception.  Unit tests reproduce all three cases.

No commit or Kaggle submission has been made.
