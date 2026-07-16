# Dataset audit

## Recovered replay layouts

- ZIPs: 20
- Full replay files: 2058
- Singular `replay/`: 164
- Newly recovered plural `replays/`: 1894
- Usable trajectories: 2074
- Decisions: 95254
- Legal candidates: 1097481

## Seat resolution

- `team_name_exact`: 1830
- `source_manifest`: 164
- `modal_target_deck_unique`: 45
- `self_play_exact_same_deck`: 30
- `team_alias_plus_modal_deck`: 3
- `alias_self_play_same_modal_deck`: 2

Ambiguous seats are excluded rather than guessed.

## Leakage controls

- Policy features use only the acting observation and a supplied legal candidate.
- Opponent private hand identities, initial full decks, outcome, future logs, and visualize data are not policy features.
- Initial decks and outcome are used only for deck clustering, sample weights, and audit reports.
- Labels are the exact legal-option indices serialized on the same seat at replay step t+1.
