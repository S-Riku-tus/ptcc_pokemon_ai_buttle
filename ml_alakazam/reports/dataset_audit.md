# Dataset audit

## Scope

- ZIP bundles inspected: 21 (ranks 1-20 plus the latest rank-1 refresh)
- Episodes catalogued: 2163
- Episodes identified as Alakazam by exact deck or repository deck evidence: 964
- Full replay episodes: 164
- Usable normal Alakazam episodes: 162
- Ranks 21-50 were not present in the workspace.

## Alignment

- Selected: `observation[t] -> action[t+1]`
- Reason: next-step legal rate 0.541444 exceeds same-step 0.417955 on normal usable episodes

- Next-step non-empty action legality: 100.00%

## Exclusions

- `replay_missing_observation_action`: 1956
- `duplicate_episode`: 43
- `abnormal_end`: 2

## Alakazam teams and variants

- Rank 1 `Majkel1337`: 164 episodes, `majkel_exact`
- Rank 2 `Majkel1337`: 100 episodes, `majkel_exact`
- Rank 3 `Yushin Ito`: 100 episodes, `majkel_near_1`
- Rank 4 `bono`: 100 episodes, `majkel_near_1`
- Rank 7 `LiamK`: 100 episodes, `alakazam_no_enriching_boss_fez_no_shaymin_dunsparce4-3_candy4_hammer4`
- Rank 9 `Rmy`: 100 episodes, `majkel_near_1`
- Rank 15 `THIRD PTCG Club`: 100 episodes, `majkel_near_3`
- Rank 16 `matsurih`: 100 episodes, `majkel_exact`
- Rank 17 `ei ei ei yikuso`: 100 episodes, `majkel_near_1`

## Leakage controls

- Policy features read only the acting observation.
- Opponent card identities are read only from Active/Bench/public zones; opponent `hand` entries are ignored and only `handCount` is used.
- Replay visualize frames and initial full decks are used only for manifest metadata, never policy features.
- Outcome and future logs affect labels/teacher weights only, never policy inputs.
- Splits are assigned by episode; no episode spans multiple time splits.

## Gate 1

Gate 1 passed for 162 episodes and 11438 decisions. Other top-team event-only bundles remain audit evidence but are not silently treated as training data.
