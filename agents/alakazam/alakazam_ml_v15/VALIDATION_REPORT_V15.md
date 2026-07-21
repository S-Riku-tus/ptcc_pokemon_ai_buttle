# Validation report v15

## Promotion gates

- v15 vs v11: 1,069-931 across two independent 1,000-game runs.
- v15 vs original v14: 549-451.
- Generic gate: 733/800 wins (91.625%).
- Generic breakdown: Grimmsnarl 87.5%, Crustle 100%, Kangaskhan 99%,
  Mega Starmie 80%.
- Zero crashes, illegal selections, policy fallbacks, and recorder exceptions.
- Meaningful attack offered with END selected: zero in measured gates.

## Static gates

- Deck must exactly equal v14: 60 cards, one ACE SPEC.
- Ranker model hash remains unchanged from v14.
- Full inherited and v15 Golden suite must pass.
- Agent validator and Python compilation must pass.

The Kaggle rating remains unknown until submission. Local gates support
promotion to a separate v15 but do not guarantee a leaderboard score.
