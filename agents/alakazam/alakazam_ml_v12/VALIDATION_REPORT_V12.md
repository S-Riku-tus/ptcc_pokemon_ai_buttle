# v12 validation report

## Static and targeted checks

- 44 agent-local tests passed, including 10 new v12 scenarios.
- The new scenarios cover Teleport role ordering, Fezandipiti survival, the
  mandatory one-Bench case, Articuno's Basic-only scope, Boss escape targets,
  gradual alternate-attacker funding, promotion/retreat, Cruel Arrow target
  priority, blocked Powerful Hand, and Dunsparce's no-cycle guard.
- Python compilation and static agent validation pass.

## Fair seat-swapped v12 vs v11

The champion-challenger harness ran 200 games with 100 games in each seat:

| Metric | v12 | v11 |
|---|---:|---:|
| Wins | 116 | 84 |
| Win rate | 58.0% | 42.0% |
| Alakazam attacks/game | 4.00 | 3.62 |
| Attack-opportunity conversion | 100% | 100% |
| Deck-out rate | 7.0% | 15.5% |
| Board-out rate | 8.0% | 9.5% |

The v12 win-rate Wilson 95% interval was 51.1%-64.6%. There were no crashes,
illegal selections, or timeouts.

The formal harness verdict remained `REJECT` because its absolute deck-out
(maximum 5%) and board-out (maximum 5%) thresholds were missed. This is kept
explicit: v12 improved both figures relative to v11 in that run, but it does
not satisfy every global promotion threshold.

## Generic matchup regression gate

At 100 games per matchup with the same seeds:

| Opponent | v12 | v11 |
|---|---:|---:|
| Grimmsnarl | 89 | 92 |
| Crustle | 100 | 100 |
| Kangaskhan | 98 | 99 |
| Mega Starmie | 80 | 83 |
| **Total** | **367/400 (91.75%)** | **374/400 (93.5%)** |

This is a small generic-pool cost, consistent with deliberately replacing the
old Kadabra/Alakazam Teleport promotion. The requested safety behavior is not
silently reverted to optimize this synthetic pool.

## Team Rocket regression gate

A 200-game local run used the Team Rocket Spidops/Mewtwo deck recovered from a
saved public replay (Articuno, Tarountula/Spidops, Mewtwo ex, and Mimikyu):

| Agent | Wins | Win rate |
|---|---:|---:|
| v12 | 163/200 | 81.5% |
| v11 | 158/200 | 79.0% |

This opponent uses the generic local pilot, so the result verifies legal route
execution and regression direction rather than claiming ladder strength.

## Decision

Keep v12 as a validated targeted challenger. It implements both requested
behaviors, passes safety checks, improves the focused Team Rocket gate, and
wins the fair v11 head-to-head. Fresh Kaggle logs are still required before
claiming that the changes improve rating stability.
