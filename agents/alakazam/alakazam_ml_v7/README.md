# alakazam_ml_v7

v7 combines the v6 deterministic safety fixes and deck with the top21-40
teacher corpus and a newly trained LightGBM ranker.

## Runtime policy

- `fallback_v3.py` remains authoritative.
- The default runtime is shadow-only and always returns the fallback action.
- A future controlled override experiment is limited to same-role Abra or
  Dunsparce bench choices.
- Attack and evolution returned to deterministic fallback because their Top-1
  accuracy declined on the new-teacher future holdout.
- Boss, Ability, Trainer, Energy, Retreat, and the other strategic actions
  remain rule-only.
- `ALAKAZAM_ML_V7_ENABLE_OVERRIDE=1` is for controlled evaluation only.

## Training corpus

The v7.1 ranker uses the v6 corpus plus ten top21-40 Alakazam rank archives
and the explicitly requested rank-1 Majkel1337 bundle.
The combined rank archives contain sixteen distinct submissions; the ML input
reader resolves metadata per submission sub-bundle.

- 18 ZIPs / 23 submissions / 16 teams / 5 deck clusters
- 6,663 full replay files / 6,562 expert trajectories
- 322,167 aligned decisions
- 3,717,660 legal candidate rows
- 6 unresolved decisions; alignment rate 99.998%

Of the 164 rank-1 replays, 120 overlapped the existing submission corpus and
were deduplicated; 44 trajectories were genuinely new.

The top21-40 addition contributed 3,249 episodes, 3,265 trajectories, 148,845
decisions, and 1,726,889 candidate rows.

On 34,336 future decisions from the added submissions, Top-1 improved by 0.45
percentage points. Bench improved by 2.58 points, while attack declined by 1.27
and evolution by 0.89. A fixed old-teacher submission holdout also improved for
bench and regressed for evolution, so live scope was narrowed to bench only.

## Deck

v7 retains the v6 list and its last-body Dudunsparce and value-gated Boss
invariants. A rank-2 teacher list ablation replaced one Dudunsparce and Max Rod
with Enriching Energy and Shaymin. It lost the 200-game seat-swapped A/B against
v6, 89-111 (44.5%), with no safety errors, so that deck change was rejected.

## Live evaluation

- rank-2 teacher deck vs v6: 89-111, rejected; v6 deck restored
- bench-only override vs v6 shadow: 102-98, 51.0%, rejected
- default shadow vs non-ML v3: 124-76, 62.0%
- all three 200-game runs had zero crashes, illegal actions, and timeouts

The old formal gate counted every engine turn on which the agent made any
selection. A replay audit showed that this was not an attack-opportunity rate:
Majkel1337 scored only 35.35% by that definition despite a 79.14% replay win
rate. The gate now keeps that number as a legacy diagnostic and instead checks
attack-opportunity conversion (minimum 95%) plus MAIN-only idle turns after the
first attack (maximum 2.0 in losses).

In fresh 200-game measurements, default-shadow v7 converted 100% of offered
attack turns with zero attackable END choices. It still remains a development
shadow agent because the v6 mirror had 11% deckout and 5.5% boardout, while the
v3 matchup had 10% boardout. Those are resource and board-continuity failures,
not failures to choose an available attack.

v7.1 adds 12 public-state features that distinguish a ready Active Alakazam
from a ready Bench backup. On the exact same time split, the feature ablation
improved overall Top-1 by 0.84 points, driven by END (+16.46) and retreat
(+13.99). Attack and evolution did not improve in that exact ablation, so they
remain deterministic fallback actions and live override scope is unchanged.

See `data/ml/alakazam_ml_v7/ATTACK_CONTINUITY_REPORT.md` and
`data/ml/alakazam_ml_v7_candidate/reports/attack_continuity_feature_ablation.json`
for the complete audit and replay-level evidence.
