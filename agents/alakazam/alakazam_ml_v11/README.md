# alakazam_ml_v11

v11 is a targeted recovery branch from the actually submitted `alakazam_ml_v10`.
It uses the 59 public v10 games, the saved eight-team Alakazam corpus, the
2026-07-20 leaderboard snapshot, and the attached independent analysis.

The deck is unchanged. The top Alakazam lists share a 54-card core. v11
therefore isolates policy changes before another deck experiment.

## Main policy changes

- If both Abra and Dunsparce are in the opening hand, start Dunsparce and keep
  Abra as the evolution route.
- Rare Candy is promoted above Kadabra only when Candy creates the first
  same-turn attack or, after Alakazam's three-card draw, an immediate KO.
- Rare Candy selects a fueled Active Abra; Kadabra on the Bench receives an
  expected-value bonus when drawing two can find Candy for an Active Abra.
- A low-deck draw for Psychic Energy is allowed only at deck eight or lower,
  at least 50% estimated hit probability, no usable Psychic in hand, and real
  next-turn loss pressure. Ordinary low-deck filtering remains blocked.
- Fezandipiti ex receives Energy only for a same-turn pivot or a concrete prize
  route that is at most one attachment away.
- Shaymin and non-mirror Xerosic plays require visible, actionable threats.

## ML decision

See `CHANGELOG_V11.md` and `VALIDATION_REPORT_V11.md`.
