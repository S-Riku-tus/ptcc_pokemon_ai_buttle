# v10 changelog

## Evidence

- Refreshed the current Kaggle top-40 snapshot on 2026-07-19; rank 40 was
  already above 1034, so a stable 900 rating is an intermediate milestone.
- Audited 96 public v9 ladder games: all offered attacks were taken, while 9
  losses were deck-outs and 7 were board-outs.
- Reused the 16-team Alakazam expert corpus and treated v9 ladder losses as
  failure evidence, not imitation labels.

## Policy

- Added one-Energy escape sequencing for one-retreat-cost support Active
  Pokemon when a powered benched Alakazam can attack immediately.
- Preserved v9 Hammer/Mist, Fezandipiti mode, dual-Kadabra, Boss, and attack
  rules.
- Kept the Shaymin + Max Rod deck after controlled deck-only ablations.

## ML

- Added 19 reconstructed observation/action interaction features (274 total).
- Added bounded submission, team, and action-frequency balancing.
- Retrained and distilled a new 50-tree LightGBM ranker.
- Added semantic-equivalence filtering so interchangeable card copies cannot be
  reported as meaningful overrides.
- Kept all strategic live authority with the deterministic policy because the
  four holdouts were mixed.
