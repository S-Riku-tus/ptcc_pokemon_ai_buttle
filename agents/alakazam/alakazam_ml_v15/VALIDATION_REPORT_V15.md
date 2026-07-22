# Validation report v15

## Candidate promotion gates

- Exact candidate vs v11: 1,069-931 across two independent 1,000-game runs.
- Exact candidate vs original v14: 549-451.
- Generic gate: 733/800 wins (91.625%).
- Generic breakdown: Grimmsnarl 87.5%, Crustle 100%, Kangaskhan 99%,
  Mega Starmie 80%.

## Finalized-agent rerun

- v15 vs v11 with v14's deck forced on both sides: 508-492 (50.8%).
- v15 vs unchanged v14: 556-444 (55.6%).
- Generic gate: 734/800 wins (91.75%): Grimmsnarl 91%, Crustle 100%,
  Kangaskhan 97.5%, Mega Starmie 78.5%.
- The finalized runtime selected ML on 1.7-2.4% of decisions, confirming the
  scope is live but narrow.
- Zero crashes, illegal selections, policy fallbacks, and recorder exceptions.
- Meaningful attack offered with END selected: zero across all 2,800 final
  rerun games.

## Static gates

- Deck must exactly equal v14: 60 cards, one ACE SPEC.
- Ranker model hash remains unchanged from v14.
- All 58 inherited and v15 Golden tests pass.
- Agent validator and Python compilation must pass.

The Kaggle rating remains unknown until submission. Local gates support
promotion to a separate v15 but do not guarantee a leaderboard score.
