# Grimmsnarl ML v6 — the Froslass over-evolve was never a modelling error

Date: 2026-08-06
Parent: `grimmsnarl_ml_v5` (v5.1: ladder 963.7 over 66 games, 44-22 public, all
six pre-registered resource targets met)
Model: byte-identical to v5, SHA256 `dabc1589…b93f79`
Runs analysed: `data/runs/grimmsnarl/20260806_grimmsnarl_ml_v5_sub55275642`
(66 games), plus the 4,097-game same-deck corpus behind v5.1's ranker

**Headline.** v5 takes the Froslass evolve on 59 of the 59 turns it is offered.
Three versions have tried to move a MAIN preference with feature columns and
failed. The reason is not in the features and not in the planner: **the pilot the
ranker is pinned to takes it on 95.7% of their own offered turns, the second
highest rate of the 21 pilots in the corpus.** v5 is copying an outlier
faithfully. So v6 changes the pin — for that one decision class, and for nothing
else. The Froslass evolve drops to 46 of 59 turns (78.0%) and 11 of 25
net-negative boards (44.0%), at a cost of **18 changed decisions in 6,095**, with
every other monitored behaviour identical to v5 and 0 errors.

## 1. The measurement that reframes the problem

No previous version had measured what each pilot does *per pilot* — only per
rating band, which averages the thing that matters. Per team, over their own
games in the frozen 4,097-game selection
(`measure_teacher_by_gap.py` → `teacher_by_gap.json`):

| team | score | games | Froslass /turn | mirror | net-neg | Froslass /decision | dead Stamp | attach when legal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16371703 | 1220.2 | 268 | 0.805 | **0.526** | 0.694 | **0.284** | 0.507 | 0.879 |
| 16422241 | 1151.0 | 300 | 0.846 | 0.692 | 0.647 | 0.496 | 0.495 | 0.847 |
| 16463316 | 1141.3 | 300 | 0.996 | 0.976 | 0.987 | 0.816 | 0.420 | 0.787 |
| 16561259 | 1126.3 | 112 | 0.726 | 0.321 | 0.677 | 0.372 | **0.330** | 0.760 |
| 16531269 | 1121.6 | 300 | 0.951 | 0.844 | 0.894 | 0.786 | 0.441 | **0.893** |
| … | | | | | | | | |
| **16494330 (our pin)** | 1077.6 | 161 | **0.957** | 0.806 | 0.844 | 0.721 | **0.713** | 0.835 |
| field, weighted | — | 4,097 | 0.913 | 0.742 | 0.834 | 0.599 | 0.560 | 0.823 |
| field range | — | — | 0.726–0.996 | 0.321–1.000 | 0.647–1.000 | 0.284–0.829 | 0.330–0.869 | 0.751–0.893 |
| **v5, ladder** | 963.7 | 66 | **1.000** | — | 0.760 | 0.747 | 0.750 | 0.728 |

Two things follow, and the second one is uncomfortable.

**The pin is the field's worst teacher on two of the three open gaps.** 0.957 on
the Froslass evolve is second of 21; 0.713 on taking an unplayable Unfair Stamp
is the worst of 21 against a weighted field rate of 0.560. Every "our agent is
off the end of the field distribution" finding in the v4 and v5 reports is
therefore, in part, a statement about *whose policy we asked for*.

**No pilot is good at everything.** 16371703 has the lowest Froslass decision
rate and a good attachment rate but is the least imitable pilot on the deck.
16531269 has the best attachment rate and a bad Froslass rate. 16463316 is rank
3 and evolves Froslass on 99.6% of offered turns. A single global pin cannot be
chosen to fix a set of gaps, which is exactly why v6 selects the pilot **per
decision class** instead of once for the agent.

## 2. Does the Froslass gap actually run with pilot rating? Partly not.

Every version in this line has justified a change with "the behaviour is
monotone in pilot rating and we are off the end of it". That is a claim about 21
points, and `measure_gap_rating_order.py` tests it as one — Spearman rho with a
20,000-shuffle two-sided permutation p:

| behaviour | rho vs collected score | p | rho vs current score (n=9) | p |
|---|---:|---:|---:|---:|
| Froslass /turn | −0.355 | 0.116 | +0.133 | 0.743 |
| Froslass /turn, mirror | −0.337 | 0.137 | +0.201 | 0.601 |
| Froslass /turn, net-negative | −0.373 | 0.099 | −0.033 | 0.949 |
| Froslass /decision | −0.066 | 0.775 | +0.233 | 0.557 |
| **dead Unfair Stamp taken** | **−0.626** | **0.0029** | −0.467 | 0.210 |
| attachment made when legal | +0.233 | 0.312 | +0.017 | 0.981 |

**The Froslass rating gradient is not significant.** The direction is right and
consistent across three of its four framings, but at n=21 it does not clear
noise, and on the *current* leaderboard the sign flips. The Unfair Stamp gap is
the one that survives the test at p = 0.003.

So the honest case for v6 is not "the winners refuse the evolve". It is:

1. v5 plays a hard **100.0%** where the most extreme of 21 pilots plays 99.6%
   and the weighted field plays 91.3%. A policy pinned at the boundary of the
   observed range has no state-sensitivity left to lose.
2. On the boards where Freezing Shroud puts more counters on our side than
   theirs, v5 evolves on 76.0% of decisions. That is an argument from the card
   text, not from a correlation: the evolve hands over damage for nothing.
3. The measured cost is 18 decisions in 6,095 with no regression anywhere else,
   so the change is close to free even if the upside is small.

That is a variance-reduction bet, and it is stated as one.

## 3. What the class is, exactly

`measure_escalation_scope.py`, over the same 66 games:

| | |
|---|---:|
| all decisions | 6,347 |
| single-pick MAIN decisions | 2,755 |
| MAIN decisions offering a Froslass evolve | **83** |
| share of all decisions | **1.31%** |
| share of MAIN decisions | 3.01% |
| mean options when it fires | 9.02 |
| … of which also offer a Grimmsnarl ex evolve | **11 of 83** |
| … also offer a Dark Energy attachment | 41 of 83 |
| … also offer an attack | 38 of 83 |

The runtime's own counter independently reports `escalation_offered = 83` on the
same run, which is the cross-check that the trigger column
(`evolve_froslass`) selects the class the scope script defines.

Because class mode can only change a decision inside the class, **1.31% is a
hard upper bound on v6's change rate**, not an estimate. The pre-registered
budget of "under 2% of all decisions" is satisfied by construction.

## 4. The counterfactual probe, on v5's own boards

`scripts/analyze_grimmsnarl_v3_behaviour.py` replays the 66 stored games
decision by decision, asks the candidate at every select and advances the game
with the action v5 actually took, so every column below is measured on identical
states. The replay *is* v5, so `agreement` is literally "share of decisions
unchanged from what v5 played", and 0.9811 for v5 against itself is the
instrument's own floor (the probe skips multi-pick selects, so a handful of
intra-turn history columns drift).

| | v5 (baseline) | **v6 (shipped)** | global pin 16371703 |
|---|---:|---:|---:|
| decisions | 6,095 | 6,095 | 6,095 |
| agreement with v5's play | 0.9811 | **0.9782** | 0.8643 |
| decisions changed beyond the floor | — | **18** | 712 |
| escalated / scored / moved | — | 83 / 83 / **16** | — |
| **Froslass evolve, own turns** | 58/59 = 98.3% | **46/59 = 78.0%** | 46/59 = 78.0% |
| **Froslass evolve, per decision** | 62/83 = 74.7% | **49/83 = 59.0%** | 49/83 = 59.0% |
| **… net-negative shroud ledger** | 19/25 = 76.0% | **11/25 = 44.0%** | 11/25 = 44.0% |
| Grimmsnarl ex evolve, own turns | 81/130 = 62.3% | **81/130 = 62.3%** | 56/130 = 43.1% |
| Dark attachment, own turns | 230/316 = 72.8% | **230/316 = 72.8%** | 248/316 = 78.5% |
| enabling attachment, own turns | 186/195 = 95.4% | **186/195 = 95.4%** | 189/195 = 96.9% |
| Boss played, own turns | 42/118 = 35.6% | **41/118 = 34.7%** | 28/118 = 23.7% |
| Adrena-Brain passes damaged Grimmsnarl | 92/256 = 35.9% | **92/256 = 35.9%** | 89/256 = 34.8% |
| Bench-30 best-prize KO chosen | 52/52 = 100% | **52/52 = 100%** | 51/52 = 98.1% |
| counters moved = maximum | 385/385 | **385/385** | 385/385 |
| feature / score errors | 0 | **0** | 0 |

**The narrow escalation reproduces the global elite pin's Froslass numbers
exactly — 46/59, 49/83 and 11/25 are identical — while changing 18 decisions
instead of 712 and keeping every other behaviour at the v5 value.** The global
pin is what the v5 report named as the natural v6 and it is refused here on
evidence: it drops the Grimmsnarl ex evolve by 19.2 points and the Boss rate by
11.9 on a deck whose whole plan is a fuelled Grimmsnarl ex, and it is the only
configuration measured that loses a best-prize Bench-30 kill.

It also shows what the escalation does *not* buy. The global pin is the one
configuration that moves the attachment rate (72.8% → 78.5%, field 82.3%), and
the class escalation leaves it exactly where v5 had it. The attachment gap is
reachable through the pin, but not through this class.

### 4b. All 18 changed decisions, individually

v6's entire footprint is 18 decisions, so it is listed rather than summarised
(`behaviour_v6_shipped.json` → `groups.froslass_evolve.changed_decisions`). What
v5 played → what v6 plays:

| v5 | v6 | n |
|---|---|---:|
| evolve Froslass | ability (Adrena-Brain / Punk Up) | 6 |
| evolve Froslass | evolve Morgrem or Grimmsnarl ex | 2 |
| evolve Froslass | attach a Dark Energy | 2 |
| evolve Froslass | end the turn | 2 |
| evolve Froslass | bench a Munkidori | 1 |
| evolve Froslass | play Petrel | 1 |
| evolve Froslass | **attack** | 1 |
| ability | play Petrel | 1 |
| bench an Impidimp | evolve Morgrem | 1 |
| play Buddy-Buddy Poffin | attach a Dark Energy | 1 |

Two things this settles that no aggregate rate could:

- **No change moves away from an attack.** One moves *to* one. The pre-registered
  "lethal-attack take rate ≥ 97%" gate therefore cannot have moved: losing a
  lethal attack requires choosing a non-attack where the pin chose an attack,
  and that appears zero times in the whole footprint.
- The five changes where v5 was *not* evolving Froslass — the cost of scoring
  the class rather than vetoing the action — are ability→Petrel,
  bench-Impidimp→evolve-Morgrem and Poffin→attach. None gives up a prize and two
  of the three are the direction the field's own rates point.

## 5. Teacher and mode were both chosen by this probe, not by rating

Four candidates on the same 66 games. Every row is class mode except the last.

| escalation teacher | score | model's Top-1 on that pilot | Froslass /turn | net-neg | moved |
|---|---:|---:|---:|---:|---:|
| 16371703 | 1220.2 | 0.797 | **0.780** | **0.440** | 16 |
| 16561259 | 1126.3 | 0.839 | 0.831 | 0.520 | 13 |
| 16422241 | 1151.0 | 0.813 | 0.881 | 0.480 | 14 |
| 16371703, `confirm` mode | 1220.2 | 0.797 | 0.780 | 0.440 | 13 |

16371703 moves the behaviour furthest and lands in the middle of the
pre-registered 60–85% band, so it ships. The lower-rated, more imitable pilots
move it less; on this class the teacher's own rate dominates the fidelity
difference, which is the opposite of what
[[grimmsnarl-imitability-vs-rating]] would predict for a whole policy and is
worth knowing before the next class is chosen.

`confirm` mode — the pin still decides, and the escalation pilot is asked only
when the pin's own argmax *is* the evolve — produces **identical** Froslass
numbers from 13 changed decisions instead of 16. It is kept as the documented
control (`GRIMMSNARL_ESCALATION=confirm`) rather than shipped, because class
mode is a teacher choice rather than a one-directional veto: it can also *take*
the evolve on boards where the escalation pilot would and the pin would not,
which is what keeps the class state-sensitive instead of monotonically
suppressed. On these 66 games that freedom cost 3 extra changed decisions and
bought nothing measurable, so the choice is a design one and is recorded as
such.

## 6. Measured and deliberately **not** shipped

Same discipline as v4, which refuted two of its five inherited priorities, and
v5, which refuted three.

### The Petrel → Unfair Stamp class — the best evidence, held back

This is the gap that survives the rating test (rho −0.626, p = 0.0029) and where
the pin is the worst of 21 pilots (0.713 against a weighted field 0.560). The
class is implemented and measured (`GRIMMSNARL_ESCALATION_CLASSES=petrel_stamp`):
39 selects offer an Unfair Stamp across the 66 games, the escalation moves 4 of
them and all 4 are refusals of the Stamp, agreement 0.9808. That is about a
third of the gap, on 39 decisions.

It is not shipped because one ladder run must measure one change, and because it
is the smaller effect on the boards we have. It is pre-registered as the v7
class, with the escalation teacher to be chosen from 16561259 (0.330), 16463316
(0.420) or 16531269 (0.441) rather than automatically 16371703 (0.507).

### A base-model retrain — refused on arithmetic, twice

The user's question was whether to also strengthen the ranker. Both available
routes were measured and neither is worth a run *for v6*:

1. **More trees.** v5.1's own sweep has 2,000 → 2,500 trees at +0.0005 Top-1
   for +34% export size and 3,515 at +0.0016 for 79.2 MB and ~88 ms/move
   against the deployed 45.1 MB and 38.6 ms/move. Training stopped at the
   4,000-round cap with `best_iteration = 3515`, so the curve had not flattened
   — but the export budget had.
2. **Raising fidelity to the escalation pilot.** The obvious targeted retrain is
   `--focus-team 16371703`, since v6's behaviour on the class *is* the model's
   reproduction of that pilot. The class is 1.31% of decisions. Moving that
   pilot's Top-1 by an optimistic 3 points would change 3% of 1.31% = **0.04%
   of decisions**, while any weight on one pilot is measured to cost pooled
   Top-1. The arithmetic kills it.

Keeping the model byte-identical is also what makes the next ladder run
interpretable: v6 minus v5 is one conditional pin and nothing else.

### But the *data* refresh trigger is now met, and for a reason worth reading

`measure_refresh_opportunity.py` took a fresh top-60 leaderboard snapshot,
downloaded one replay per representative submission and hashed the deck. Of the
current top 40:

- **6 submissions still play the exact 60-card list** (`9714ab5c3996f6cc`); 33
  play another archetype and 1 plays a different Grimmsnarl list.
- 4 of those 6 are not in the frozen selection — 2 new teams (16541765 rank 25,
  16606656 rank 31) and 2 new submissions from known teams (**16561259 rank 10 /
  1111.0 with 303 episodes**, 16531269 rank 30) — for **559 new same-deck games
  available**, a ~13.6% corpus increase.
- Only **8 of our 21 corpus teams are still in the top 60 at all**. Both the
  deployed pin (16494330) and the escalation teacher (16371703) have dropped
  out, and rank-3 16463316 has switched off this deck entirely.

The archive itself shows zero new games since 2026-08-05, which reads like a
static meta and is not one: the EpisodeService only serves episodes per
submission id, and the submissions our pilots were tracked under have stopped
playing. See [[kaggle-teacher-log-refetch]].

That clears two of the documented rebuild triggers (≥200–500 new same-deck
games; ≥100 new games for a strategically important top team) and a third
qualitatively (a material metagame shift). It is the right next training run —
as its own gated candidate, after v6's ladder result, not bundled into it.

## 7. Validation

- **159 agent tests pass** — 144 inherited from v5 unchanged, plus 15 new: what
  the escalation fires on, that a non-MAIN select with the trigger column never
  escalates, that `last_scores` carries one pilot's scores and never a mixture,
  that `off` mode is v5, that the dense team code is the corpus's code for team
  16371703 (so a rebuilt corpus fails the test instead of silently scoring a
  different pilot), and that the shipped class list is Froslass only.
- v5's 144 tests still pass unchanged in v5's own directory.
- `scripts/validate_agent.py` passes with no warnings.
- 0 feature errors, 0 score errors, 0 illegal selects across the probes.
- The planner's counters are unchanged on the same run (`heal_overrides` 15,
  `froslass_overrides` 0, `punk_alloc_*` 0), i.e. the escalation did not start
  handing the planner boards it wants to override.
- Local paired arena, 60 alternating-seat mirror games against v5 on seed 1705:
  **29-31 (48.3%)**, 0 crashes, 0 illegal selects, 0 draws, 45.06 ms/move
  against v5's 43.08 (+4.6%, the second scoring pass on 1.3% of decisions plus
  the diagnostic pass). Read that for safety and timing only: 299 paired mirror
  games could not separate v5 from v4, and the mirror is symmetric in the
  behaviour this change is about — both sides refuse the same evolves.
- Local arena against `alakazam_ml_v35`, the matchup where the v5 ladder run went
  6-7 on 13 games, both agents on the same seed 314159 and 40 games:
  **v6 24-16 (60.0%) against v5's 22-18 (55.0%)**, 0 crashes and 0 illegal
  selects on both. Two games on 40 is not a signal, but it is not a regression
  either, and it is the only cross-archetype evidence available before a ladder
  run.

### 7b. The firing rate is higher in live play than on stored boards

The probe is teacher-forced, so it visits the boards *v5* created.
`measure_live_escalation.py` plays v6 for real — 24 games against
`alakazam_ml_v35`, 14-10, 0 feature or score errors:

| | teacher-forced on v5's boards | live |
|---|---:|---:|
| escalated / scored decisions | 83/5,456 = **1.52%** | 46/1,649 = **2.79%** |
| moved / escalated | 16/83 = 19.3% | 8/46 = 17.4% |

The share nearly doubles, and the mechanism is specific to this change:
refusing the evolve leaves the Snorunt on the bench with the Froslass still in
hand, so the class re-fires on later turns. The *fraction* the escalation then
moves is stable (19.3% vs 17.4%), which is the number that says the class
behaves the same on v6's own boards. Ladder gate 3 below is set on the live
figure, not the probe's.

## 8. What to check on the next ladder run — stated before it

Behavioural targets 1–4 are already met on the 66 stored boards; the run is to
confirm they hold against the live field and to see whether they convert.

1. Froslass evolve on **60–85%** of offered own turns (v5: 100%).
2. Froslass evolve on **≤ 55%** of net-negative-shroud decisions (v5: 76.0%).
3. `escalation_offered` between **1.5% and 4.0%** of *scored* decisions
   (`ml.main_decisions`), and `escalation_moved / escalation_offered` between
   **10% and 30%**. Live play measured 2.79% and 17.4%; the teacher-forced probe
   measured 1.52% and 19.3%, and the band spans both because refusing the evolve
   keeps the class alive for later turns. Outside that band the deployed class is
   not the class this report measured, and the behaviour numbers below do not
   transfer.
4. Unchanged, to be watched for regression: Grimmsnarl ex evolve per turn
   (62.3%), Dark attachment per turn (72.8%), enabling attachment (95.4%),
   Boss per turn (35.6%), Adrena-Brain damaged-Grimmsnarl pass (35.9%),
   best-prize Bench-30 (100%), counters-moved-maximum (100%).
5. Every v5 resource gate holds, since none of that code changed: Punk Up mean
   2.5–2.8 and five-card searches under 5%, Darkness left in deck ≥ 4.0 overall
   and ≥ 2.4 from turn 5, Adrena-Brain ≥ 6.0 a game.
6. 0 crashes, 0 illegal selects, 0 timeouts.

Outcome reading, per the v5 analysis and [[kaggle-ladder-rating-noise]]:
**pool at least two independent submissions and 200+ games**, rank by
performance rating against opponents' initial ratings rather than by final
rating, and report win rate by opponent band, by seat and by opponent
archetype. Do not read the Alakazam matchup below ~40 games: v4 went 11-2 and
v5 6-7 on 13 games each.

One caution specific to this run: the same snapshot that justifies the next data
refresh says the mirror is thinning at the top — 6 of the current top 40 on this
deck against 51% of the top 50 on 2026-08-02 — so the mirror share of the field
will not match v5's 21 of 66, and a mirror-conditioned comparison across runs
needs the per-archetype split rather than the aggregate.

## 9. The route after this

The two levers this report leaves on the table are both larger than v6:

1. **The class table.** The mechanism generalises: pick the pilot per decision
   class by measurement. The Petrel class is implemented and waiting; the
   attachment gap is measured to be reachable through the pin (72.8% → 78.5%
   under the global elite pin) but has no narrow class, since a Dark Energy
   attachment is offered on 316 of the run's own turns rather than 83 decisions.
2. **Outcome learning.** Nothing in this report escapes imitation. The pin, the
   class and the teacher are all still answers to "what would pilot X do", and
   the ceiling of that question is pilot X. [[grimmsnarl-imitation-saturated]]
   named branching the top candidates through the engine and learning the prize
   or win difference; the escalation makes that easier to aim, because it
   identifies the decision classes where the imitation prior is demonstrably
   the wrong one to follow.
