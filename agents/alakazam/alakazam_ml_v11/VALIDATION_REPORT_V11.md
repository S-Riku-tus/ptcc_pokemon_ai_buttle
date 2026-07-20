# v11 validation report

## Targeted checks

34 tests cover the inherited tactical guards plus the new opening, Rare Candy,
Kadabra draw-out, low-deck Energy gamble, Fezandipiti, Shaymin, and feature
logic. Static agent validation reports a legal 60-card deck. No tested battle
produced a crash or illegal selection.

## Mirror gate against submitted v10

Four seeds, 200 games per seed:

| Seed | v11 | v10 |
|---:|---:|---:|
| 741 | 106 | 94 |
| 742 | 101 | 99 |
| 743 | 107 | 93 |
| 744 | 100 | 100 |
| **Total** | **414** | **386** |

v11 won 51.75%. This is a regression gate, not evidence that Kaggle rating
will rise by a particular amount.

After the final low-deck activation integration, a 400-game four-seed smoke
gate finished 209-191 (52.25%) with zero crashes and zero illegal selections.

## Generic matchup gate

Using the same local generic policies and 200 games per matchup:

| Opponent | v11 | v10, same seed |
|---|---:|---:|
| Grimmsnarl | 178/200 | 174/200 |
| Crustle | 200/200 | 200/200 |
| Kangaskhan | 195/200 | 188/200 |
| Mega Starmie | 150/200 | 156/200 |
| **Total** | **723/800 (90.38%)** | **718/800 (89.75%)** |

## Decision

Accept the targeted policy changes and unchanged deck. Reject the earlier
broad backup/runway prototype. Keep ML shadow-only until the new feature set is
trained on fresh high-quality teacher replays and clears all holdouts plus a
live battle gate.
