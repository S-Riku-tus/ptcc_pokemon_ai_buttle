# marnies_grimmsnarl_ex_v3

Logic-only upgrade of the identical 60-card leaderboard Grimmsnarl list.  v3 keeps
v2's 908-replay route/target model and retunes it for the prize race against
hand-scaling opponents.

## What changed vs v2

- **Unfair Stamp as pre-attack hand denial** — played as a free Item before the
  attack, scaled to the opponent's hand size, to strip a large hand (the fuel for
  hand-size damage/draw engines) and then still attack.  This is the validated
  driver of the win-rate gain.
- **Second Munkidori vs a racing opponent** — widens the spread/heal engine;
  measured neutral vs the tested Alakazam agents, kept for wider-field value.
- **Rejected:** Froslass gating and an explicit heal-the-attacker bias were tested
  and hurt, so they were dropped.

## Runtime files

- `main.py`
- `policy_base.py`
- `deck.csv`

## Results (self-play, large-sample; see `VALIDATION_REPORT.md`)

- vs `alakazam_ml_v11`: 31.8% -> **34.7%** (+2.9pt, 4000 games each)
- vs `alakazam_ml_v10`: 38.6% -> **43.0%** (+4.4pt, 2000 games each)

The bundled engine is not seed-reproducible, so results are reported over large
samples with 95% confidence intervals.

## Evidence base (deck / route model, inherited from v2)

- Submission 54837973: 161 analyzable replays
- Submission 54797424: 747 analyzable replays
- Combined: 908 replays

See `STRATEGY.md`, `CHANGELOG.md`, and `VALIDATION_REPORT.md`.
