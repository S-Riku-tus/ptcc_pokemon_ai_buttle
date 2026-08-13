# v12 validation report

## Static and targeted checks

- 44 agent-local tests passed, including 10 new v12 scenarios.
- The new scenarios cover Teleport role ordering, Fezandipiti survival, the
  mandatory one-Bench case, Articuno's Basic-only scope, Boss escape targets,
  gradual alternate-attacker funding, promotion/retreat, Cruel Arrow target
  priority, blocked Powerful Hand, and Dunsparce's no-cycle guard.
- Python compilation and static agent validation pass.

## Decision

Keep v12 as a validated targeted challenger. It implements both requested
behaviors and passes static safety checks. Fresh Kaggle logs are still required
before claiming that the changes improve rating stability.
