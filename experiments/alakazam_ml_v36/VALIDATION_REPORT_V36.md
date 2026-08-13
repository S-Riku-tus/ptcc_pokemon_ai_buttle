# Alakazam ML v36 validation report

Date: 2026-08-02  
Decision: **do not create or promote a v36 agent**  
Requested target: strict Top-1 agreement `> 0.90`  
Result: **target not met**

## Bottom line

The latest teacher submission was still active. I incrementally recovered 244
previously unseen Yushin Ito games and rebuilt the corpus from 2,512 unique
episodes (125,127 decisions and 1,109,057 candidate rows). This removes the
earlier assumption that no fresh teacher data was available.

It does not change the promotion decision. The first prospective evaluation
on all 164 newest test episodes scored 80.24% strict Top-1. An online split
then used the first 80 new episodes for training, the next 64 for validation,
and the final 100 as a shadow test. The validation-selected final refit on
train+validation scored 80.85% (3,305/4,088; Wilson 95% interval
79.61-82.02%). No tested configuration approached 90%.

The 100-episode online shadow block had already appeared inside the initial
164-episode prospective score before the online experiment was designed. It
was never used for fitting or hyperparameter selection, but it is not a
single-touch test. The 80.85% number is therefore supporting evidence, not a
claim that a new untouched 90% result exists. The first 80.24% prospective
result is the cleanest estimate from the new cohort.

## Rating-1000 diagnosis

Kaggle EpisodeService returned 67 completed public games for v35 submission
55166624 and 71 for v33 submission 55129390. The local v35 archive has one
additional replay, giving 41-27 over 68 games; rating buckets use API games
with an opponent `initialScore`.

| Opponent initial rating | v33 | v35 |
|---|---:|---:|
| below 700 | 3/3 | 6/6 |
| 700-799 | 10/11 | 12/12 |
| 800-899 | 19/32 | 18/32 |
| 900-949 | 7/16 | 2/10 |
| 950-999 | 3/6 | 2/5 |
| 1000+ | 0/3 | 0/2 |
| **900+ pooled** | **10/25 (40.0%)** | **4/17 (23.5%)** |

The Wilson intervals overlap because these are small samples, but the result
supports the original concern. v35's 59.7% API win rate was supported by a
perfect 18-0 below rating 800, not by performance against stronger teams. Its
maximum observed updated score was 911.27 versus 928.21 for v33.

## Evaluation data

The combined index contains five same-teacher cohorts: 180, 115, 994, 980,
and 244 trajectories. Duplicate seats were removed before extraction. The
newest cohort spans episode IDs 89286150 through 89468065 and contributes
9,958 decisions.

Two chronological views were used:

1. Prospective refresh: 2,268 old episodes for training, the first 80 new
   episodes for validation, and the last 164 new episodes for test.
2. Online adaptation: 2,344 episodes for training (including the first 80 new
   episodes), 64 for validation, and 100 for the shadow test. Episode overlap
   between all three splits is zero. After configuration selection, the final
   refit used train+validation, or 2,408 episodes, while retaining the last
   100 episodes for scoring.

The new states differ materially from the old corpus. Trainer decisions rose
from 19.03% of old training decisions to 22.91% in the first new training
block, Hammer rose from 1.35% to 2.26%, and Ability fell from 14.74% to
12.00%. This is state-distribution and metagame drift even though the teacher
submission itself is unchanged.

## Experiment results

Older frozen-block experiments remain useful for rejecting model families,
but they are separated from the new prospective evaluation below.

| Candidate | Validation Top-1 | Test/Shadow Top-1 | Decision |
|---|---:|---:|---|
| v34/v35 ranker, old frozen split | 83.12% | 83.01% | reference |
| decision-level action hierarchy, old split | **83.65%** | 81.32% | reject: temporal overfit |
| empirical pair precedence, old split | 83.12% | 83.01% | reject: validation selected zero weight |
| conditional pairwise Top-3, old split | 83.12% | 83.01% | reject: validation selected zero weight |
| old configuration on 164 wholly new test episodes | 81.86% | 80.24% | prospective target failed |
| newest 25% of online training episodes | 77.23% | 76.42% | reject: inadequate state coverage |
| all online training history | **81.11%** | 80.55% | selected online configuration |
| Top-4 LambdaRank + 4x new-cohort weight | 80.25% | 80.28% | reject: validation worsened |
| selected configuration refit on train+validation | n/a | **80.85%** | final shadow estimate; target failed |

The old two-stage K=5 cascade reached 83.48% on the old test, but validation
separated K=3 and K=5 by one decision while their test results differed by
1.26 points. It was rejected as unstable. Binary labels, alternate label
gains, seed averaging, more trees, fixed action order, and turn-history
features had already failed to provide a stable step change in v33-v35.

## Why strict 90% is not currently attainable

On the online full-history model, strict Top-1 was 80.55%, Top-3 was 97.68%,
and order-insensitive turn-set agreement was 93.44%. Its 4,088 shadow-test
decisions break down as:

- 3,293 correct;
- 430 recoverable same-turn ordering errors;
- 97 premature attack/end ordering errors; and
- 268 genuine divergences.

Strictly exceeding 90% requires at least 3,680 correct decisions, or 387 more
than the full-history model. Even if divergence and premature errors remain
unchanged, that requires recovering 387 of the 430 recoverable ordering
errors (90.0%) without breaking a correct choice.

Those errors do not follow a global order. Both Evolve-before-Energy and
Energy-before-Evolve occur, Trainer-versus-Trainer is the largest single
reversal, and mistakes are spread over turn positions 0 through 15. The
conditioned pairwise model, action hierarchy, Top-K cascade, explicit
intra-turn features, and priority tables cover the main post-ranking model
families; none produced a stable validation gain. Reaching 90% therefore
requires learning the teacher's latent turn plan, not another small ranking
override.

The current information bottleneck is that only 80 new-policy-distribution
episodes were available for adaptation before validation. Keeping only recent
episodes destroyed state coverage, while adding all history left the new
ordering distribution underrepresented. The exact replay memory is not a
solution either: its earlier episode-held-out coverage was only about 1.7%.

## Promotion decision and next required information

No `alakazam_ml_v36` agent directory was created and no v35 runtime/model file
was overwritten. The best defensible policy remains v35 until a challenger
passes both of these gates:

1. strict Top-1 above 90% on a newly collected, single-touch chronological
   test cohort; and
2. a frozen high-rating matchup suite that improves 900+ performance,
   especially Dragapult and Grimmsnarl, without regressing the mirror.

The next useful collection should contain substantially more games from the
same active teacher distribution (preferably at least roughly 1,000 before
reserving validation and test), or access to the teacher policy for active
state querying. More old games or mixed teachers are not substitutes: older
data already showed negative transfer, and direct multi-teacher training
conflicted on the Yushin holdout.

For the rating-1000 objective, strict imitation alone is also insufficient.
v33 beat 900+ opponents more often than v35 despite the later imitation
pipeline. The next policy should optimize outcome against high-rating matchup
states after the imitation model supplies a safe prior, rather than treating
harmless same-turn action order as the sole objective.

## Main artifacts

- `teacher_index.csv` / `teacher_index.summary.json`: deduplicated 2,512-game teacher index
- `corpus_report.json`: 125,127-decision extraction report
- `online_split_report.json`: chronological split and zero-overlap audit
- `v36_refresh_training_report.json`: first prospective new-cohort evaluation
- `online_full_residual.json`: online full-history score and error taxonomy
- `online_ordering_errors.json`: action-pair and turn-position diagnostics
- `online_final_refit_report.json`: validation-selected train+validation refit
- `../alakazam_ml_v35/ladder_opponent_buckets_v33_v35.json`: rating-conditioned live results
- `../alakazam_ml_v35/ladder_strategy_55166624.json`: 68-replay matchup audit
