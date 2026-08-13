# Grimmsnarl ML v7 — imitation candidates + counterfactual value

## Policy structure

1. The byte-identical v6 ranker generates candidate actions.
2. The v6 arithmetic planner keeps its dominance/safety authority.
3. From turn 5, at most once per turn, the top 3 MAIN candidates within 3.0
   ranker points are branched in the real engine.
4. Each branch follows the v6 policy to terminal or the next player's turn.
5. A 381-column public state model may override v6 only for a predicted win
   probability gain of at least 0.06.
6. Any model/search/rollout error returns the v6 action.

Attack and END are now comparable because the value model excludes private
hand identities and current-player-only flags before normalising the leaf back
to our original seat.  Candidate and chosen-action columns never enter the
value head.

## Value-model evidence

The frozen source is the current-top-four exact-list corpus: 1,014 selected
team/seat relations, 991 unique episodes and 76,947 decision states.  One
state row is retained per decision and every episode receives equal total
training weight.

| chronological split | states | episodes | AUC | log loss |
|---|---:|---:|---:|---:|
| validation | 8,567 | 118 | 0.8451 | 0.4964 |
| test | 9,495 | 118 | 0.7849 | 0.5414 |

The deployment cutoff follows the turn-band test rather than the pooled AUC:

| test band | AUC |
|---|---:|
| turns 1–4 | 0.6299 |
| turns 5–8 | 0.8139 |
| turns 9+ | 0.8582 |

This is why value search is disabled before turn 5.  The complete fit and
calibration table are in `training_report.json`.

## Imitation-fidelity gate

The full new runtime was teacher-forced over the pinned pilot's 18
chronological held-out games.  It matched 1,559 of 1,706 single-pick decisions:
**91.38% exact raw-index agreement**, above the pre-registered 90% gate.

The search layer actually ran in this replay:

- 70 searches and 180 branches;
- 1 value override;
- 0 branch errors and 0 incomplete branches;
- 0 ranker feature/score errors and 0 planner errors.

The old Petrel v7 measured 91.27%.  Removing that pin and adding conservative
value search therefore did not trade away the high-fidelity behaviour which
made v4–v6 strong.

## Remaining uncertainty

- Hidden opponent zones use a deterministic mirror-deck prior.  This is legal
  and adequate for most own-turn effects, but it is not an archetype belief
  model.
- The state model predicts outcomes observationally; the engine supplies the
  counterfactual transition, but not a randomised action label.
- The next evidence should be a Kaggle challenger run while v6 remains the
  champion/control.  Promotion should require both rating and opponent-band
  evidence, not the headline win rate alone.

## Artifacts

No Kaggle submission was made.
