# Grimmsnarl ML v11 RC1

## Decision

## What changed

v11 is v8 plus one once-per-turn full-turn arithmetic search.  v8 supplies the
root argmax, up to three semantically distinct candidates and every continuation
decision.  Each candidate is advanced with cabt's real Search API to the end of
our turn.  The leaf compares:

* game result and both players' prize changes;
* whether an attack completed and observable damage landed;
* active/backup Grimmsnarl readiness and active survival margin;
* board-out safety, evolution progress and useful versus stranded Darkness;
* retained known hand plans and deck runway.

Search uses two legal hidden-state determinizations.  The second is evaluated
only for candidates that strictly improve in the first.  An override requires
a strict, non-regressive improvement in both.  Ties, conflicting samples,
missing Search state, incomplete branches and engine errors return v8.

The ranker, deck, fallback policy, features and existing arithmetic planner are
byte-identical to v8.  Only `main.py`, compact ranker-state snapshot support and
the new `arithmetic_search.py` layer differ.

## Verification

### Static and regression

* 172 v11 tests passed, covering every inherited v3-v8 policy invariant and
  the new search safety conditions.
* `validate_agent.py` passed: 60 cards, 19 unique cards, no warnings.
* Python compilation passed for all changed runtime/evaluation modules.
* Submission builder/validator/loader isolation tests passed (9/9).
* SHA-256 equality with v8 was verified for `deck.csv`, `fallback_policy.py`,
  `ml_features.py`, `ml_planner.py`, `policy_base.py` and the 45 MB ranker.
* Ranker SHA-256 remains
  `dabc15894cae4ebf49ab6fa6d91e7af0ad81b2c88751da5ad2cb05a326b93f79`.
* The built 19-entry submission archive is 10,939,740 bytes with SHA-256
  `36d42d4f0386619c193405066315611b77581105ecf1b86a8b8f95a7be1644e9`.
  Its extracted copy loaded the ranker and search layer with no errors and
  completed a two-game package smoke with no crash or illegal selection.

The repository-wide test command also ran.  Its remaining 2 failures and 10
setup errors are pre-existing missing-fixture failures for
`alakazam_ml_v2_expanded` and the Spidops reconstruction package; none imports
or exercises v11.

### Alakazam-going-second counterfactual

The first ten field replays through turn 8 contain 676 decisions.  v11 changed
7 (1.04%).  Five of the seven selected an action/card that the successful
teacher played later in the same turn, confirming that the new signal mostly
addresses the known sequence-order error rather than replacing end-of-turn
plans.  The two genuinely teacher-external changes were:

* Petrel before Munkidori: both determinizations ended with one extra ready
  Grimmsnarl, +2 evolution progress and no prize/attack regression;
* Rare Candy before Munkidori: both ended with +2 evolution progress while
  preserving three prizes and the attack.

Strict all-context Top-1 moved from 0.8933 to 0.8859 because ordering a later
teacher action first is counted wrong.  On MAIN decisions, v8/v11 strict scores
were 286/334 and 281/334; same-turn order-insensitive scores were 334/334 and
332/334.  This is a safety diagnostic, not an outcome estimate.

## Ladder gate

Submit v11 and keep v8 active as the rollback baseline.  Do not interpret the
first few games.  The promotion target is at least 120-150 games, with the
first 10-15 calibration games reported separately, at least 60-80 games versus
1000+ opponents, and all of the following:

1. no crash/illegal-action signal and acceptable overage time;
2. last-50 median rating at least 1100;
3. positive win-rate delta versus v8 in matched opponent-rating bands;
4. no collapse in Alakazam-going-second or another family accounting for at
   least 5% of the observed meta.

If those fail, v8 remains champion and the recorded override ledger identifies
which arithmetic class to remove or retrain.  If they pass, v11 becomes the new
training policy for a DAgger/residual iteration, breaking the 1077.6 teacher
ceiling with labels produced from v11's own reached states.
