# Feature specification

The policy is a legal-candidate ranker. Every row combines one acting-observation state vector with one legal candidate vector.

## State features

- `turn`
- `go_first`
- `your_prizes`
- `opp_prizes`
- `your_deck`
- `opp_deck`
- `your_hand`
- `opp_hand`
- `your_bench`
- `opp_bench`
- `supporter_played`
- `energy_attached`
- `retreated`
- `stadium_played`
- `stadium_id`
- `your_active_id`
- `your_active_hp`
- `your_active_max_hp`
- `your_active_damage`
- `your_active_energy`
- `opp_active_id`
- `opp_active_hp`
- `opp_active_max_hp`
- `opp_active_damage`
- `opp_active_energy`
- `your_alakazam`
- `your_kadabra`
- `your_abra`
- `your_dudunsparce`
- `your_dunsparce`
- `opp_board_signature`
- `your_board_signature`
- `low_deck`
- `hand_alakazam`
- `hand_kadabra`
- `hand_abra`
- `hand_candy`
- `hand_boss`
- `hand_hammer`
- `hand_xerosic`
- `hand_energy`
- `select_type_code`
- `context_code`
- `option_count`
- `min_count`
- `max_count`

## Candidate features

- `candidate_index`
- `option_type_code`
- `card_id`
- `attack_id`
- `skill_id`
- `target_card_id`
- `player_index`
- `source_area`
- `target_area`
- `source_index`
- `target_index`
- `candidate_hp`
- `candidate_energy`
- `hand_delta`
- `board_delta`
- `attack_damage`
- `ko_possible`
- `is_end`
- `is_attack`
- `is_ability`
- `is_retreat`
- `is_evolve`
- `is_energy`
- `is_boss`
- `is_hammer`
- `is_xerosic`
- `is_candy`
- `is_dudunsparce`
- `high_importance`
- `context_option_code`

## Explicit exclusions

Opponent private hand card IDs, unrevealed deck order, prizes, future draws, post-action state, and final outcome are not policy features.
A legal empty selection (`minCount=0`) is represented by a `NONE` pseudo-candidate and converted back to `[]` at runtime.
