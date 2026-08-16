# Dragapult ML v2 (typed-route imitation)

The exact-list Dragapult imitation pilot, rebuilt around the defect that
decided v1's ladder run. Not a search agent.

## What v1 got wrong

Phantom Dive costs one Fire and one Psychic. v1 described an attachment by the
target's card id and its *total* energy count, so "Fire onto a Dragapult
holding Psychic" and "Fire onto a Dragapult holding Fire" were the same row.
The model could not express the distinction, and on the ladder it showed:

| | v1.0 live (22 public games) | same-deck teachers |
|---|---:|---:|
| duplicate-colour attachments / game | 0.870 | 0.037 |
| completes a colour pair / game | 1.043 | 1.701 |
| Dragapult evolved onto a 2-colour body / game | 0.174 | 0.809 |
| first Phantom Dive, own-turn mean | 6.5 | 4.0 |
| games that ever used Phantom Dive | 63.6% | 94.0% |
| own turns per game | 8.05 | 6.86 |
| win rate | 0.500 | 0.651 |

The action *rates* already matched the teachers (attach 0.894 vs 0.892 per own
turn, evolve Drakloak 0.958 vs 0.954, Adrena-Brain 0.926 vs 0.903, Recon
Directive 0.838 vs 0.823). Only the argument of the action was wrong.

v1.1 answered this with a broad deterministic override of energy, evolution,
retreat, Boss and search. On the held-out split that override seized 2,322
decisions and agreed with the teachers on 40.2% of them, against the model's
own 72.7%: it fixed one case and made everything it touched worse.

## What v2 changes

1. **Typed route features.** Per-body Fire/Psychic/Dark counts, route ETA
   before and after the candidate action, "completes the pair", "arms Phantom
   Dive", and the same three columns computed against Crispin's context card.
2. **Resolved card tables** (`scripts/build_dragapult_card_tables.py`): prize
   value, weakness/resistance, printed HP, the attack-cost frontier, and the
   bodies that prevent all damage from a Pokémon ex. Phantom Dive's 200 is not
   flat — Crustle, Sylveon and Cornerstone Ogerpon take zero from it.
3. **Ability options resolved correctly.** `OptionType.ABILITY` carries
   `area`/`index`, not `inPlayArea`/`inPlayIndex`. v1 read the wrong pair in
   both the features and the rule policy, so Recon Directive and Adrena-Brain
   were invisible to the model and indistinguishable to the policy.
4. **`remainingOverageTime` removed.** It was v1's 15th highest-gain column and
   is an artefact of whose machine ran the game: teacher logs run 572–592 s,
   ours run 591–599 s.
5. **Spread placement reads the remaining counter budget** instead of assuming
   all six are still in hand.
6. **Ultra Ball's discard ordered by the teachers' measured rate**, generated
   from the training split only (`scripts/build_dragapult_discard_table.py`).
   It is a multi-pick select, so the ranker never sees it.
7. **Corpus refreshed**: 1,392 verified trajectories from 15 teachers, up from
   854 from 9. The 1,000-trajectory data gate now passes.
8. **The broad override is gone.** What remains is one mechanical guard: a
   duplicate-colour route attachment is replaced when the same decision offers
   one that completes the pair. On the 16,380-decision test split it bound
   0 times — the model no longer needs it. `DRAGAPULT_GUARD_DISABLE=1` removes
   even that, so its contribution stays measurable.

## Results

Controlled feature ablation — identical 854 episodes, identical splits,
patience 800 on both arms:

| | v1 features | v2 features |
|---|---:|---:|
| test Top-1 | 0.7568 | 0.7606 |
| test Top-3 | 0.9656 | 0.9668 |
| damage-counter context | 0.7268 | 0.7650 |
| Crispin attach-to | 0.7935 | 0.8261 |

Raising early stopping from 200 to 800 was worth +0.0011 on its own, so the
gain is the features, not the tree count.

Submitted-shell agreement on the 167-episode / 16,380-decision chronological
test split:

| | v1.1 | v2 |
|---|---:|---:|
| semantic agreement | 0.6862 | **0.7295** |
| MAIN | 0.5657 | 0.6391 |
| Ultra Ball discard | 0.2100 | 0.4484 |
| multi-pick slice | 0.5387 | 0.6491 |
| duplicate route attachments | 0 (forced by the guard) | 2 of 681 |
| legal rate / exceptions | 1.0 / 0 | 1.0 / 0 |
| mean / p95 latency | 10.6 / 30.2 ms | 14.9 / 43.1 ms |

The teachers themselves make 8 duplicate route attachments in 614 on the same
episodes, so v2 is at the teacher rate without being forced there.

### Counterfactual on the decisions that actually lost the run

Teacher agreement says v2 imitates better on held-out teacher games. It does not
say v2 fixes the decisions v1.0 got wrong on the ladder. Replaying all 23
downloaded episodes and forcing every agent onto the trajectory that was really
played (`scripts/counterfactual_dragapult_v2.py`), over the 1,955 single-pick
decisions of the run:

| | submitted v1.0 | v2 |
|---|---:|---:|
| duplicate-colour route attachments chosen | 21 | **0** |
| pair-completing route attachments chosen | 24 | **109** |
| agreement with the action v1.0 actually played | 0.9898 | 0.8271 |

At the 12 decisions where v1.0 attached a duplicate colour *while the same
decision offered an attachment that completes the Fire+Psychic pair*, v2 takes
the completing attachment 11 times, repeats the duplicate 0 times, and once
plays something else entirely. v2 still reproduces 82.7% of v1.0's live
actions, so this is the same policy with the route argument corrected, not a
different agent.

Reproduction commands are in `experiments/dragapult_ml_v2/README.md`.

## What this does not prove

Agreement and behaviour are fidelity measurements, not a rating. The exact-list
pilots this model imitates sit at 1048–1153; the modified lists above them are
piloted better, not built better (team 16380946 scored 1229.3 on this list and
1224.0 after changing it). A ladder run is the only test that settles v2.
