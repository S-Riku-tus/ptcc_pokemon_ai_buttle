# Grimmsnarl ML v27: design and validation

## Decision

v27 keeps the v22 deck and default ranker. It does not promote the higher
teacher-match v25 model to the default policy. It instead combines four narrow,
measured components:

1. v22 as the default action policy.
2. The v24 same-deck-mirror Froslass guard.
3. The v26 public wall and imminent deck-out guards.
4. A corrected, adaptive belief search used only in public exact-list mirrors.

This is a conservative submission candidate, not a claim that offline teacher
agreement is an adequate ladder objective.

## Why v26 looked weak

The first 15 v26 ladder games were 9-6, but the split was 8-0 going first and
1-6 going second. Fourteen games were outside the exact mirror scope in which
v26 can invoke H2. The only mirror was a win. This sample therefore does not
identify H2 as the source of the apparent regression. Its shape is more
consistent with an early, highly turn-order- and matchup-skewed sample.

Higher imitation accuracy also need not improve win rate. The v25 metric asks
whether a model reproduces one teacher action on states in the logged teacher
distribution. It does not measure counterfactual win probability, error
recovery after the model visits its own states, matchup calibration, or whether
several nearly equivalent legal actions have different labels. A stronger
teacher can therefore yield a more accurate clone that is still less robust
than v22 on the ladder.

The concrete v26 search defect was hidden-state construction. Opponent hand,
deck, and prizes could be populated by independently repeating the deck list,
so one simulated world could contain impossible duplicate counts. Extending
that model blindly to a deeper horizon would compound model error.

## Adaptive H3

The v27 search subtracts every public card from a 60-card multiset, including
evolution stacks, attached Energy, tools, stadiums, known hand cards, and known
prizes. Hidden hand, prizes, and deck are then allocated without replacement
from the one remaining pool. Any inconsistent observation fails closed to v22.

Search starts with three H2 worlds. A clear, safe H2 winner can be accepted as
before. H3 is considered only when H2 is ambiguous, the public-state confidence
proxy is at least 0.55, and the state is tactically important. It evaluates v22
against one challenger and expands 3 -> 5 -> 7 -> 9 worlds only while the result
remains promising and the episode budget can afford it. Rule-based and v25
opponent responses alternate across worlds. No override is allowed if even one
world is worse than v22, if prize/deck-out safety regresses, or if the mean
value gain is below 0.04. H3 requires at least five completed worlds to accept.

Thus v27 can read farther than v26, but depth is purchased only where the public
information and decision ambiguity justify it.

## Validation

- 232 unit and regression tests pass.
- Strict multiset tests verify exact 60-card conservation for both players and
  rejection of an impossible fifth copy.
- Synthetic adaptive tests cover H3 activation, 3 -> 5 -> 7 expansion, and
  immediate fallback when one hidden world is worse.
- A real Search API mirror state completed two direct H3 branches in 4.969 s.
- The same real state completed six adaptive H2 branches in 9.069 s and rejected
  H3 because belief confidence was 0.545, demonstrating the intended gate.
- Teacher-forced replay covered 1,269 decisions: 38 targeted wall-cell changes,
  zero changes across 77 ordinary control decisions, and zero changes across
  19 mirror decisions when search was disabled.

These checks establish implementation safety and scope. They do not establish
ladder superiority; that needs a larger, turn-order-balanced submission sample.

## Reports

- `replay_probe.json`
- `search_engine_probe.json`
