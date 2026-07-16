# Final report — expanded Alakazam imitation ranker

## Conclusion

The original project design was retained: exact legal-option imitation learning, episode-level leakage control, a distilled dependency-free tree runtime, and a deterministic fallback. The critical parser bug was fixed by supporting both `replay/episode_*.json` and `replays/episode_*.json`.

The expanded corpus improves broad generalization, but Boss, Retreat, Xerosic, and Hammer remain unsafe for direct ML control. They are therefore hard-routed to the v12 fallback; Energy is ML-controlled only at high confidence.

## Data recovery

- Full replay count: 2058
- Recovered plural-path replays: 1894
- Usable trajectories: 2074
- Decisions: 95254
- Candidate rows: 1097481
- Teams / submissions / decks: 19 / 20 / 8

## Deck clusters

- `majkel_exact` distance 0.0: 567 trajectories, Larry | Majkel1337 | matsurih | me and the lads
- `majkel_near` distance 1.0: 803 trajectories, 213tubo | Bulba-Zero | Rmy | Yushin Ito | bono | ebisu_ya | lolzpo smw | soyukke
- `majkel_near` distance 1.0: 101 trajectories, ei ei ei yikuso
- `majkel_near` distance 3.0: 101 trajectories, THIRD PTCG Club
- `alakazam_variant` distance 5.0: 202 trajectories, Ebi | LiamK
- `alakazam_variant` distance 5.0: 101 trajectories, capbloo
- `alakazam_variant` distance 9.0: 100 trajectories, 5.5
- `alakazam_variant` distance 10.0: 99 trajectories, rick & shikitora

## Holdout evaluation

| Holdout | Top 1 | Top 3 | MRR | ECE | Fallback |
|---|---:|---:|---:|---:|---:|
| time_holdout | 52.36% | 80.88% | 0.684 | 0.113 | 68.09% |
| team_holdout | 46.44% | 79.49% | 0.647 | 0.146 | 65.13% |
| submission_holdout | 56.67% | 83.07% | 0.711 | 0.161 | 71.82% |
| deck_holdout | 50.57% | 79.73% | 0.670 | 0.111 | 68.26% |

## Weight ablation

- `full`: Top1 52.30%, Top3 80.49%, MRR 0.682
- `uniform`: Top1 52.68%, Top3 80.69%, MRR 0.684
- `no_deck_distance`: Top1 52.37%, Top3 80.60%, MRR 0.683
- `no_rank_outcome`: Top1 52.38%, Top3 80.67%, MRR 0.683

## Singular-only versus expanded corpus

- `legacy_singular_only`: Top1 61.60%, Top3 86.18%, MRR 0.752
- `expanded_all_teams`: Top1 66.69%, Top3 85.83%, MRR 0.777

## Runtime safety policy

- Default probability threshold: 0.55
- Margin threshold: 0.12
- Boss / Retreat / Xerosic / Hammer: always fallback
- Energy: ML only when probability is at least 0.85 and the margin gate also passes
- A fallback-confirmed immediate KO is never overridden
- Dudunsparce self-removal is blocked when it would leave no body or a critically low deck
- Nested target/search selections and multi-select decisions remain fallback-controlled

## Validation

- Replay-policy smoke illegal actions: not rerun
- Replay-policy smoke exceptions: not rerun
- Distilled runtime supports both numeric and LightGBM categorical tree splits.
- Actual Kaggle Rating improvement is not claimed without official-engine ladder evaluation.

## Remaining risks

- The rank49 Jack replay bundle was not included.
- Focus-action expert weighting improved rare actions but materially reduced global Top1, so it was rejected.
- Deck/rank/outcome weighting was only weakly supported by ablation; weights are intentionally mild.
- Battle smoke requires the repository's official `vendor/cg` and opponent agents.
