# Grimmsnarl ML v13: design and pre-submit evidence

## Decision

v13 is not a further tuning of v12's arithmetic search.  It is a first-stage
sticky mixture of policies:

- v8 for the default field and the mirror;
- v9's unconditioned current-top-four model for publicly identified Alakazam;
- the deterministic wall-aware fallback for Dwebble, Crustle, Cornerstone Mask
  Ogerpon ex, Sylveon, or Neutralization Zone;
- v8's narrow proof-based planner only; no broad whole-turn search.

The default route may make one irreversible safety promotion to the wall state
machine if the opponent initially looks generic and only reveals the wall line
later.  Alakazam and mirror routes do not switch experts after they lock.

## Why

The ladder evidence is a crossover, not a monotone improvement.  Reported win
rates moved from v8 to v9/v12 as follows:

| matchup | v8 | v9 | v12 |
| --- | ---: | ---: | ---: |
| Grimmsnarl mirror | 68.8% | 57.1% | 52.2% |
| Alakazam | 36.4% | 64.7% | 72.2% |

Therefore one global replacement must throw away evidence from one side of the
table.  v12 also produced the concrete no-progress wall failure in episode
91279034.  Its objective rewarded attacking, local damage and bodies, which
can prefer a locally legible leaf while breaking the multi-turn prize route.

## Submit gate

The first ladder submission should be treated as a route experiment.  Promote
only if the mirror stays compatible with v8, Alakazam stays compatible with or
above v9, and wall logs contain no repeated no-progress cycle.  Aggregate
rating alone is insufficient; report route, matchup, first player, attack turn,
Froslass timing, wall attacks and expert-load diagnostics for every episode.

The next learned version should train a dedicated matchup-conditioned
Alakazam expert or route head.  It should not restore broad search until a
multi-turn objective and off-policy evaluation prove that the search preserves
the selected route.
