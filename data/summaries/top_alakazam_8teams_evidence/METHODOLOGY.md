# Methodology

- Input: eight uploaded ZIP bundles.
- Scope: only `EPISODE_TYPE_PUBLIC`; validation/self-play episodes were excluded.
- Total: 793 public games (Yushin Ito 100; the other seven teams 99 each).
- Source within each replay: `steps[0][0].visualize`, whose frames contain non-duplicated semantic logs (`TurnStart`, `Play`, `Evolve`, `Attach`, `Attack`, `MoveCard`, etc.) and full board states.
- Deck lists were reconstructed from frame 0's complete 60-card action lists.
- Dudunsparce cycle = Dudunsparce moving from Active/Bench back to Deck.
- “Manual switch heuristic” excludes Switch events immediately following an attack, but can still include card/effect-driven switching; treat it as directional, not exact retreat count.
- Matchup labels are derived from the opponent's most-used attackers/evolutions, then merged into broad archetype groups.
- Raw log win rate is not leaderboard rating. Opponent rating and opponent mix differ, so same-deck team comparisons are suggestive rather than controlled A/B tests.
- Card-use conditional win rates are selection-biased; they describe timing/context, not causal card value.
