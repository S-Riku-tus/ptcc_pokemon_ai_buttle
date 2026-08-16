# Dragapult ML v3 (board-width imitation)

v3 keeps v2's exact 60-card list and typed-Energy imitation policy.  It targets
the remaining ladder failure tail: games in which the first Dragapult appears,
but there is no second route to sustain Phantom Dive.

## Changes from v2

- Adds observation-only board-width features: line bodies in play, open bench
  slots, backup route ETA, missing Drakloak/Dragapult pieces, visible upper
  bounds on unseen line cards, and whether a candidate search or evolution
  adds/advances a backup attacker.
- Weights five observable board-width/continuity state groups by 1.5 during
  training.  Whole LambdaRank groups are weighted; individual candidate labels
  are not.
- Adds a bounded Active-evolution guard.  It rejects evolving a zero-Energy
  Active Drakloak when no legal attachment can enable Jet Headbutt and selects
  the next-best safe ranked action.  A measured survival exception allows the
  evolution when it turns a Phantom Dive knockout into survival.
- Keeps the OOD gate unchanged.  Relaxing opponent card-ID support was measured
  on all 54 live triggers and changed only seven choices, concentrated damage
  less effectively, and had no demonstrated value.
- Keeps teacher pin 16380946.  On the exact v22 Grimmsnarl list this teacher was
  8-0 with 3.75 Phantom Dives and 2.50 Dragapult created per game.  Repinning
  to 16445292 reduced every v22 development metric in a 40-game screen.

All new inputs are derived from the current public observation.  Hidden prize
identities, opponent hand/deck contents, future state, result, and environment
timers are not features.

## Training

The corpus contains 1,392 exact-list seat trajectories from 15 teachers,
121,816 ranking decisions, 732,375 candidate rows, and 470 corpus features.
`teacher_team_id` is appended at fit/export time for 471 deployed features.
Splits are chronological within each teacher; episodes and teachers receive
equal base loss mass.

```powershell
.\.venv\Scripts\python.exe scripts\train_grimmsnarl_v2_teacher.py `
  --corpus data\ml\dragapult_v3\corpus_full.npz `
  --team-feature --split-mode per-team `
  --episode-equal-weight --teacher-equal-weight `
  --hard-state-set dragapult_v3 --hard-state-weight 1.5 `
  --num-boost-round 5000 --early-stopping 300 --threads 18 `
  --output-model data\ml\dragapult_v3\ranker_board_focus.txt `
  --report experiments\dragapult_ml_v3\train_board_focus.json

.\.venv\Scripts\python.exe scripts\export_grimmsnarl_v1_model.py `
  --model data\ml\dragapult_v3\ranker_board_focus.txt `
  --corpus data\ml\dragapult_v3\corpus_full.npz `
  --teacher-team 16380946 `
  --report experiments\dragapult_ml_v3\train_board_focus.json `
  --output agents\dragapult\dragapult_ml_v3\ranker_model.json
```

Held-out matrix results: Top-1 0.7455, Top-3 0.9652, order-insensitive
Top-1 0.8879, MAIN Top-1 0.6754.  The exported model has 1,088 trees, 471
features, and is 24,597,795 bytes.

## Runtime and live-counterfactual checks

The actual submitted shell was replayed on 167 held-out episodes / 16,380
decisions: semantic agreement 0.7263 (95% Wilson interval 0.7194-0.7330),
MAIN 0.6373, legal rate 1.0, and zero agent/feature/score exceptions.  The
first timing pass measured mean 24.4 ms, p95 71.7 ms, max 156.6 ms.

The evolution guard was swept over all 2,450 single-pick decisions in the 26
downloaded v2 games.  It changes the two measured failure states:

- episode 93603391, Mega Lucario: do not donate a two-Prize, zero-Energy
  Active Dragapult;
- episode 93607548, Teal Mask Ogerpon: same failure after retreating into an
  unpowered Drakloak.

A held-out teacher counterexample (episode 93140945) exposed the necessary
survival exception: evolving a damaged Drakloak survives an opposing ready
Phantom Dive.  After adding the exception, that teacher's 14-game / 1,186
decision test block has zero guard overrides, 100% legal actions, and zero
exceptions, while both live fixes remain bound.

## Grimmsnarl v22 evaluation

Local games are used only against `agents/grimmsnarl/grimmsnarl_ml_v22`, not
as a Dragapult-mirror acceptance test.  Native shuffle is unseedable, so the
record is secondary to repeated development measurements.

| agent | games | record | mean PD | P(0/1) | P(2+) | P(4+) | max Dragapult |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 | 80 | 20-60 | 2.450 | 26.25% | 73.75% | 33.75% | 1.525 |
| v3 selected | 120 | 34-86 | 2.792 | 20.00% | 80.00% | 36.67% | 1.642 |

The opponent deck is not merely the same archetype: its exact 60-card hash
`9714ab5c3996f6cc` occurs in 151 teacher games, where Dragapult went 115-36
(76.16%).  v3 still does not beat the much stronger v22 pilot locally, so no
claim of matchup superiority is made.  It does reduce the failure tail and
increase sustained attacks relative to v2.

Rejected controls are retained under `experiments/dragapult_ml_v3`:

- hard-state weight 2.0: 21-59 over 80 v22 games, with worse P(4+) and
  held-out MAIN;
- current-teacher loss weight 1.5: better offline pin agreement but 7-33 and
  reduced board width against v22;
- pin 16445292: 8-32 and worse development;
- win-weight 1.25: no v22 record gain and weaker sustained-attack metrics;
- PD4+ episode weighting: not selected; it introduced future-conditioned
  sample selection and did not complete within the fixed model-capacity run.

No Kaggle submission or commit is performed by this directory.
