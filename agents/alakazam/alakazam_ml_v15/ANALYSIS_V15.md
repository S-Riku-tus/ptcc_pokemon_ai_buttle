# v15 logic and ML analysis

## Constraints

- `alakazam_ml_v14` remains unchanged.
- v15 uses exactly the same 60 cards as v14.
- Every candidate was tested as a separate wrapper before promotion.

## Deterministic findings

The corrected behavioral harness showed that v14 no longer selected END while
a meaningful attack was offered. The residual loss to v11 was therefore not
the original attack-completion bug.

In a 300-game diagnostic, v14 used Rich 0.79 times/game and suffered 58 actual
deckout losses. Four simple Rich restrictions all regressed over 800 games,
showing that deleting its development value was not the answer. Exact v11
component ablations were more informative:

| Candidate versus v11 | Wins / 1,000 |
|---|---:|
| v14 control | 401 |
| v11 Rich score only | 414 |
| v11 ordinary Boss score only | 441 |
| both, retaining v14 protection escape | 447 |

The combined candidate reduced 300-game terminal deckouts from 58 to 38,
post-first-attack idle turns from 1.37 to 1.22/game, and raised Alakazam attacks
from 3.57 to 3.62/game. The important Boss distinction is actual versus merely
projected Active KO: a possible later KO must not automatically suppress a
profitable immediate gust.

## ML findings

v14's model was not only shadow-only; its safe bench scope also required the
same card ID, so enabling it usually changed nothing. At the old global 0.55
threshold, 1,133 decisions produced zero selected ML actions in a smoke gate.

v15 keeps the existing weights and changes only the runtime contract:

- allow Abra/Dunsparce bench-role choice;
- allow target choice for the same evolution card;
- keep every irreversible strategic class rule-only;
- apply the calibrated 0.37 class threshold to this promoted scope.

The resulting adoption rate is about 1.4-2.0% of decisions. This small scope
was enough to clear two independent v11 gates and did not regress the four
generic matchup gates.

## Promotion evidence

| Gate | Result |
|---|---:|
| v15 vs v11, run 1 | 541-459 |
| v15 vs v11, run 2 | 528-472 |
| Combined vs v11 | 1,069-931 (53.45%) |
| v15 vs original v14 | 549-451 (54.9%) |
| Four generic opponents | 733-67 (91.625%) |

Those rows are the pre-promotion candidate gates. A fresh rerun against the
files in the finalized v15 directory produced 508-492 (50.8%) against v11 with
v14's deck forced on both sides, 556-444 (55.6%) directly against unchanged
v14, and 734-66 (91.75%) across the four generic opponents. Across those 2,800
games there were zero attackable END selections, crashes, illegal selections,
policy fallbacks, or recorder exceptions. The finalized ML adoption rate was
1.7-2.4% of decisions.

The local engine is not perfectly seed-reproducible, so ladder improvement is
not guaranteed. The repeated direction, direct v14 gate, behavioral metrics,
zero runtime errors, and non-mirror gate together justify the separate v15.
