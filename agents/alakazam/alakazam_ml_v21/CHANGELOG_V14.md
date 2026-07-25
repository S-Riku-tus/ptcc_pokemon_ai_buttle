# CHANGELOG V14

## Deck and model

- Keep the v13 60-card deck unchanged.
- Keep the existing distilled LightGBM ranker unchanged and shadow-only.
- Add `ALAKAZAM_ML_V14_ENABLE_OVERRIDE` while retaining older aliases.

## Deterministic policy

1. Give same-turn Active Alakazam fuel a score of 28000.
2. Block Enriching Energy whenever a Psychic attachment can create that attack.
3. Rank END below an offered Active Alakazam attack or enabling attachment.
4. Score protected Powerful Hand at 500: below useful play, above END.
5. Permit Boss to escape a protected Active for an unprotected Bench KO.
6. Reserve all Hammers after Mist is seen while another copy remains likely.
7. Play Hammer against attached prevention Energy anywhere, not only Active.
8. Define the missing deterministic backup ETA used by the Rich draw-stop rule.

## Diagnostics

- Add counters for Mist-memory reservations, protected attack fallbacks,
  attack-first attachments, and Boss effect-lock escapes.
- Fix the unreachable v13 Enriching-attachment diagnostic branch.
